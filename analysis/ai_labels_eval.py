"""How well does the round-3 CurbRamp validator score AI-placed (RampNet) curb ramps?

Production never sends AI-authored labels to the AI validator (SidewalkWebpage's ``LabelTable.getLabelsToValidateWithAi``
filters out the ``Ai`` role), so the validator has never been measured on them. Three deployments hold RampNet
curb-ramp labels (``sidewalk-auto-labeler``), and humans have validated a few thousand of them. This script scores the
retrained CurbRamp validator on every AI-placed CurbRamp label with at least one human Agree/Disagree vote, against the
human majority, with no new ground truth (Jon, 2026-10-27: use the existing votes only). Results in
``docs/ai-labels-2026-10.md``.

Stages (subcommands)::

    # 1. Mac: the label set from fresh rawLabels pulls (<rawlabels>/<city_id>.csv), the pano list to stage, provenance
    python analysis/ai_labels_eval.py select --rawlabels RUN/rawlabels --round3-split round3/CurbRamp/split.csv \\
        --out RUN
    # 2. klone: stage the panos (analysis/ai_labels_stage.slurm), then cut the crops exactly as data_generation.py
    #    does, from the staged store copy (analysis/ai_labels_gen.slurm runs crops + score)
    python analysis/ai_labels_eval.py crops --labels RUN/labels.csv --pano-store STAGE --out RUN
    # 3. klone: score the crops with the round-3 checkpoint (evaluate.run_inference)
    python analysis/ai_labels_eval.py score --manifest RUN/manifest.csv --checkpoint retrain-2026-10/CurbRamp/dinov2/checkpoint-4340 \\
        --out RUN/predictions.csv
    # 4. Mac: the tables and figures
    python analysis/ai_labels_eval.py report --run RUN --map round3/CurbRamp/calibration/curbramp.json \\
        --map-uncapped round3/CurbRamp/calibration/uncapped/curbramp.json --round3 round3/CurbRamp \\
        --out docs/ai-labels-2026-10

**Reference.** Per label, the majority of human Agree vs Disagree votes (``validator_type == "Human"`` only, as
``data_generation.human_vote_counts`` counts; Unsure ignored; ties dropped). Almost every validated label has exactly one
vote, so the reference is one person's opinion, and the labels reached that person through the validation queue, not
at random (selection bias; ``docs/ai-labels-2026-10.md`` states both). ``EXPERTS`` (``"<role>=<user_id>,..."``, the
``analysis/analyze.py`` form, never written into the repo) adds an expert-only reference (``data_generation.expert_verdict``).

**Geometry.** AI labels were placed on the rig-frame image, so they are rendered WITHOUT the tilt correction, exactly
as ``data_generation.py`` renders ``author == ai`` rows and as sidewalk-ai-api renders ``coord_frame == "rig"``. That
holds for every provider, so the non-GSV rows that ``data_generation.valid_geometry`` drops (their *level-frame* pose
convention is unhandled) are rendered here on purpose: no pose is needed for a rig-frame point. Human-placed labels in
these deployments (few) get the tilt correction only on GSV; on other providers they are rendered untilted and flagged
``pose_unhandled``.

**Confidence tier.** The RampNet confidence of a label is not stored on the server and the run files are not on
this machine, so the tier is derived from the campaign that submitted it (``CAMPAIGNS``: each deployment's
``send_to_ps.py`` submission records give the date and threshold of every campaign). ``core`` = the 0.55 operating
point, ``band`` = the 0.30-0.55 band shipped after sidewalk-auto-labeler#20, ``mixed`` = a re-submission at 0.30.

**Leakage.** The round-3 retrain selected with ``--include-ai-labels`` (its ``generation_run.json``), so some Vancouver
AI labels were in its train and val splits. ``select`` joins the round-3 ``split.csv`` and the report's headline
excludes train and val rows (``clean``); the test rows and the never-selected rows are unseen.

Pure helpers (vote parsing, tiers, gate metrics) need only numpy/pandas and are unit-tested in
tests/test_ai_labels_eval.py; ``crops`` and ``score`` need the training environment (environment.yml) and a GPU.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import data_generation as dg  # noqa: E402

LABEL_TYPE = "CurbRamp"
GATE = 0.92
DEFAULT_CITIES = ("vancouver-wa", "richmond-va", "laurens-ia")
SEED = 20261027

#: Submission campaigns per deployment: (first UTC date, tier). A label's tier is that of the latest campaign whose date
#: is <= its ``time_created`` date. From sidewalk-auto-labeler ``runs/<city>/*.submission.json`` (richmond, laurens) and
#: its docs/server-agree-check.md (Vancouver: one campaign at the 0.55 operating point, 2025-09-18/19, before the band
#: existed). Richmond 2026-09-24 re-submitted 14 repositioned sequences at 0.30 (``results.posfix3seq.raw.jsonl``), so
#: those labels mix both tiers.
CAMPAIGNS = {
    "vancouver-wa": [("2025-09-18", "core")],
    "richmond-va": [("2026-09-05", "core"), ("2026-09-22", "band"), ("2026-09-24", "mixed")],
    "laurens-ia": [("2026-09-17", "core"), ("2026-09-22", "band")],
}
TIERS = ("core", "band", "mixed")

LABEL_COLUMNS = [
    "label_uid", "city", "label_id", "pano_id", "author", "pano_source", "time_created", "tier", "human_agree",
    "human_disagree", "human_unsure", "n_votes", "label_class", "expert_agree", "expert_disagree", "expert_class",
    "round3_split", "clean", "severity", "tags", "pano_x", "pano_y", "pano_width", "pano_height", "camera_pitch",
    "camera_roll",
]


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vote_counts(validations_json, experts=frozenset()):
    """``(agree, disagree, unsure, expert_agree, expert_disagree)`` over human votes only.

    Votes without ``validator_type == "Human"`` (the AI validator's own) are ignored, as in
    :func:`data_generation.human_vote_counts`; Unsure is counted separately.

    >>> v = '[{"user_id":"e","validation":"Agree","validator_type":"Human"},' \\
    ...     ' {"user_id":"x","validation":"Unsure","validator_type":"Human"},' \\
    ...     ' {"user_id":"ai","validation":"Disagree","validator_type":"AI"}]'
    >>> vote_counts(v, {"e"})
    (1, 0, 1, 1, 0)
    >>> vote_counts(float("nan"))
    (0, 0, 0, 0, 0)
    """
    if not isinstance(validations_json, str) or not validations_json:
        return 0, 0, 0, 0, 0
    a = d = u = ea = ed = 0
    for v in json.loads(validations_json):
        if v.get("validator_type") != "Human":
            continue
        vote = v.get("validation")
        expert = str(v.get("user_id")) in experts
        if vote == "Agree":
            a += 1
            ea += expert
        elif vote == "Disagree":
            d += 1
            ed += expert
        elif vote == "Unsure":
            u += 1
    return a, d, u, ea, ed


def majority_class(agree, disagree):
    """``correct`` / ``incorrect`` by the human majority; ``""`` on a tie or no vote.

    >>> majority_class(2, 1), majority_class(0, 1), majority_class(1, 1), majority_class(0, 0)
    ('correct', 'incorrect', '', '')
    """
    if agree > disagree:
        return "correct"
    if disagree > agree:
        return "incorrect"
    return ""


def tier_of(city, time_created, campaigns=CAMPAIGNS):
    """The confidence tier of a label from its deployment's campaign dates; ``""`` when unknown.

    >>> tier_of("richmond-va", "2026-09-05T01:10:00Z"), tier_of("richmond-va", "2026-09-22 19:03:45.1")
    ('core', 'band')
    >>> tier_of("richmond-va", "2026-09-30"), tier_of("laurens-ia", "2026-09-01"), tier_of("nowhere", "2026-09-05")
    ('mixed', '', '')
    """
    if not isinstance(time_created, str) or len(time_created) < 10:
        return ""
    day = time_created[:10]
    tier = ""
    for first_day, name in campaigns.get(city, []):
        if day >= first_day:
            tier = name
    return tier


def wilson(k, n, z=1.96):
    """Wilson 95% interval for ``k`` of ``n``: ``(lo, hi)``; ``(nan, nan)`` when ``n`` is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def served_accuracy(api_map, verdict, confidence):
    """Serve each verdict's confidence through its class map as sidewalk-ai-api does (``api_lookup_many``)."""
    from calibration._isotonic_map import api_lookup_many
    verdict = np.asarray(verdict)
    confidence = np.asarray(confidence, dtype=float)
    out = np.zeros(len(verdict))
    for cls in ("correct", "incorrect"):
        m = verdict == cls
        if m.any():
            out[m] = api_lookup_many(api_map[cls], confidence[m])
    return out


def gate_metrics(true, verdict, served, gate=GATE):
    """What the gate lets through, per verdict class and overall.

    Returns a dict with, for ``agree`` (verdict correct), ``disagree`` (verdict incorrect) and ``all``: ``n`` (verdicts
    of that class), ``n_pass`` (served >= gate), ``coverage`` (n_pass / n), ``precision`` (share of passing verdicts
    that match the human reference, Wilson 95%), and for ``disagree`` also ``catch`` = passing Disagrees / labels the
    humans marked incorrect (the Disagree recall at the gate).

    >>> r = gate_metrics(["correct", "incorrect", "incorrect"], ["correct", "incorrect", "correct"], [0.95, 0.93, 0.5])
    >>> (r["agree"]["n_pass"], r["agree"]["precision"], r["disagree"]["catch"], r["all"]["coverage"])
    (1, 1.0, 0.5, 0.6667)
    """
    true, verdict, served = np.asarray(true), np.asarray(verdict), np.asarray(served, dtype=float)
    passed = served >= gate
    out = {}
    for name, cls in (("agree", "correct"), ("disagree", "incorrect"), ("all", None)):
        m = np.ones(len(true), bool) if cls is None else verdict == cls
        n, n_pass = int(m.sum()), int((m & passed).sum())
        hits = int((m & passed & (verdict == true)).sum())
        lo, hi = wilson(hits, n_pass)
        out[name] = {"n": n, "n_pass": n_pass, "coverage": round(n_pass / n, 4) if n else float("nan"),
                     "precision": round(hits / n_pass, 4) if n_pass else float("nan"),
                     "precision_lo": round(lo, 4), "precision_hi": round(hi, 4)}
    n_incorrect = int((true == "incorrect").sum())
    out["disagree"]["catch"] = round(out["disagree"]["n_pass"] * out["disagree"]["precision"] / n_incorrect, 4) \
        if n_incorrect and out["disagree"]["n_pass"] else (0.0 if n_incorrect else float("nan"))
    return out


def auc_block(df, n_boot=1000, seed=SEED):
    """AUC(incorrect) of ``p_incorrect`` against ``true`` with a panorama-resampled 95% interval, plus accuracy and
    balanced accuracy of the raw verdict. NaN AUC with one class."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from analysis.retrain_report import cluster_boot, ci
    y = (df.true == "incorrect").astype(int).values
    s = df.p_incorrect.values.astype(float)
    verdict = np.where(df.p_incorrect > df.p_correct, "incorrect", "correct")
    out = {"n": int(len(df)), "n_incorrect": int(y.sum()), "n_panos": int(df.pano_id.nunique()),
           "acc": round(float((verdict == df.true.values).mean()), 4) if len(df) else float("nan")}
    if 0 < y.sum() < len(y):
        out["auc"] = round(float(roc_auc_score(y, s)), 4)
        lo, hi = ci(cluster_boot(y, s, df.pano_id.values, n_boot, seed)[:, 0])
        out["auc_lo"], out["auc_hi"] = round(lo, 4), round(hi, 4)
        out["bal_acc"] = round(float(balanced_accuracy_score(y, (verdict == "incorrect").astype(int))), 4)
        # recall of incorrect at precision 0.92, without any map (the model's own ranking)
        from evaluate import recall_at_precision
        r, t = recall_at_precision(y, s, GATE)
        out["recall_incorrect_at_p92"], out["threshold_at_p92"] = round(r, 4), (round(t, 4) if t is not None else None)
    else:
        out.update({"auc": float("nan"), "auc_lo": float("nan"), "auc_hi": float("nan"), "bal_acc": float("nan"),
                    "recall_incorrect_at_p92": float("nan"), "threshold_at_p92": None})
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def fmt_ci(x, lo, hi, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# ---------------------------------------------------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------------------------------------------------

def select_labels(rawlabels_dir, cities, experts=frozenset(), round3_split=None):
    """One row per CurbRamp label with >= 1 human Agree/Disagree vote in ``cities``: AI-authored ones (the study set)
    and human-authored ones (the in-deployment comparison). Returns ``(labels_df, inputs)``."""
    frames, inputs = [], {}
    for city in cities:
        path = os.path.join(rawlabels_dir, f"{city}.csv")
        inputs[city] = {"path": path, "sha256": sha256_file(path)}
        df = pd.read_csv(path, low_memory=False, dtype={"pano_id": str, "label_id": str})
        df["city"] = city
        frames.append(df[df.label_type == LABEL_TYPE])
    df = pd.concat(frames, ignore_index=True)
    counts = df.validations.map(lambda v: vote_counts(v, experts))
    for i, col in enumerate(("human_agree", "human_disagree", "human_unsure", "expert_agree", "expert_disagree")):
        df[col] = [c[i] for c in counts]
    df["n_votes"] = df.human_agree + df.human_disagree
    df["author"] = df.user_id.map(dg.label_author)
    df["label_uid"] = df.city + ":" + df.label_id.astype(str)
    assert df.label_uid.is_unique
    df["label_class"] = [majority_class(a, d) for a, d in zip(df.human_agree, df.human_disagree)]
    df["expert_class"] = df.validations.map(lambda v: dg.expert_verdict(v, experts)) if experts else ""
    df["tier"] = [tier_of(c, t) for c, t in zip(df.city, df.time_created)]
    df.loc[df.author == "human", "tier"] = ""
    df["round3_split"] = ""
    if round3_split:
        s = pd.read_csv(round3_split, dtype=str, usecols=["label_uid", "split"])
        df["round3_split"] = df.label_uid.map(dict(zip(s.label_uid, s.split))).fillna("")
    df["clean"] = (~df.round3_split.isin(["train", "val"])).astype(int)
    voted = df[(df.n_votes > 0) & (df.label_class != "")].copy()
    ties = int(((df.n_votes > 0) & (df.label_class == "")).sum())
    voted = voted.sort_values(["city", "label_id"], kind="mergesort").reset_index(drop=True)
    for c in LABEL_COLUMNS:
        if c not in voted:
            voted[c] = ""
    summary = {"inputs": inputs, "ties_dropped": ties, "per_city": {}}
    for city, g in voted.groupby("city"):
        ai = g[g.author == "ai"]
        summary["per_city"][city] = {
            "ai_labels_total": int((df[df.city == city].author == "ai").sum()),
            "ai_voted": int(len(ai)), "ai_panos": int(ai.pano_id.nunique()),
            "ai_incorrect": int((ai.label_class == "incorrect").sum()),
            "ai_two_plus_votes": int((ai.n_votes >= 2).sum()),
            "ai_expert_decided": int((ai.expert_class != "").sum()),
            "ai_by_tier": {t: int((ai.tier == t).sum()) for t in TIERS},
            "ai_in_round3": {k: int(v) for k, v in ai.round3_split.value_counts().items() if k},
            "human_labels_total": int((df[df.city == city].author == "human").sum()),
            "human_voted": int((g.author == "human").sum()),
            "pano_source": {k: int(v) for k, v in g.pano_source.value_counts().items()},
        }
    return voted[LABEL_COLUMNS], summary


def cmd_select(args):
    experts = dg.parse_experts(os.environ.get("EXPERTS"))
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    labels, summary = select_labels(args.rawlabels, cities, experts, args.round3_split)
    os.makedirs(args.out, exist_ok=True)
    labels.to_csv(os.path.join(args.out, "labels.csv"), index=False)
    panos = sorted({f"{c}/{p[:2]}/{p}.jpg" for c, p in zip(labels.city, labels.pano_id)})
    with open(os.path.join(args.out, "stage_files.txt"), "w") as f:
        f.write("\n".join(panos) + "\n")
    summary.update({
        "selected_on": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cities": cities,
        "experts_configured": len(experts), "campaigns": CAMPAIGNS, "label_type": LABEL_TYPE,
        "round3_split": {"path": args.round3_split, "sha256": sha256_file(args.round3_split)} if args.round3_split else None,
        "n_labels": int(len(labels)), "n_panos": len(panos),
        "labels_csv_sha256": sha256_file(os.path.join(args.out, "labels.csv")),
    })
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "campaigns"}, indent=2))


# ---------------------------------------------------------------------------------------------------------------------
# crops (GPU) and score (GPU)
# ---------------------------------------------------------------------------------------------------------------------

MANIFEST_COLUMNS = ["label_uid", "city", "label_id", "pano_id", "author", "pano_source", "label_class", "pano_path",
                    "pano_native_w", "pano_native_h", "tilt_applied", "pose_unhandled", "tilt_beta", "depth_m",
                    "crop_half_px", "clipped", "crop_path", "status"]


def _num_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def cmd_crops(args):
    """Cut one crop per label from the staged store copy, exactly as data_generation.py cuts a training crop
    (``read_store_panorama`` -> ``create_perspective_crop`` -> ``save_webp``), resumable on ``manifest.csv``."""
    labels = pd.read_csv(args.labels, dtype={"label_id": str}, low_memory=False)
    if args.cities:
        labels = labels[labels.city.isin(args.cities.split(","))]
    if args.limit:
        labels = labels.head(args.limit)
    dg.check_pano_store_marker(args.pano_store)
    present = [c for c in labels.city.unique() if os.path.isdir(os.path.join(args.pano_store, c))]
    missing_cities = sorted(set(labels.city.unique()) - set(present))
    if missing_cities:
        print(f"no pano-store directory for {missing_cities}: their labels are written as no_pano")
    manifest_path = os.path.join(args.out, "manifest.csv")
    crops_dir = os.path.join(args.out, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    done = set()
    if os.path.exists(manifest_path):
        old = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        ok = old[(old.status == "ok") & old.crop_path.map(lambda p: os.path.exists(os.path.join(args.out, p)))]
        done = set(ok.label_uid)
        old[old.label_uid.isin(done)].to_csv(manifest_path, index=False)  # drop failed rows: they are retried
        print(f"resuming: {len(done)} crops already ok, {len(old) - len(done)} rows retried")
    else:
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    todo = labels[~labels.label_uid.isin(done)].sort_values(["city", "pano_id"], kind="mergesort")
    device = dg.resolve_device(args.allow_cpu)
    depth_model = dg.load_depth_model("vitb", device)
    last = (None, None, None)
    status_counts, t0 = Counter(), time.time()
    with open(manifest_path, "a", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            crop_path = f"crops/{row.city}_{row.label_id}.webp"
            human_gsv = row.author == "human" and str(row.pano_source).lower() == "gsv"
            rec = {"label_uid": row.label_uid, "city": row.city, "label_id": row.label_id, "pano_id": row.pano_id,
                   "author": row.author, "pano_source": row.pano_source, "label_class": row.label_class,
                   "pano_path": "", "pano_native_w": "", "pano_native_h": "", "tilt_applied": 0,
                   "pose_unhandled": int(row.author == "human" and not human_gsv), "tilt_beta": args.tilt_beta,
                   "depth_m": "", "crop_half_px": "", "clipped": "", "crop_path": crop_path, "status": ""}
            try:
                if last[0] != (row.city, row.pano_id):
                    if row.city in missing_cities:
                        read, info = "no_pano", {"pano_path": None}
                    else:
                        read, info = dg.read_store_panorama(args.pano_store, row.city, row.pano_id)
                    last = ((row.city, row.pano_id), read, info)
                pano_bgr, info = last[1], last[2]
                rec.update({k: ("" if v is None else v) for k, v in info.items() if k in rec})
                if isinstance(pano_bgr, str):
                    rec["status"] = pano_bgr
                else:
                    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])
                    crop, cinfo = dg.create_perspective_crop(
                        depth_model, pano_rgb, float(row.pano_x) / float(row.pano_width),
                        float(row.pano_y) / float(row.pano_height),
                        camera_pitch=_num_or_none(row.camera_pitch), camera_roll=_num_or_none(row.camera_roll),
                        apply_tilt=human_gsv, beta=args.tilt_beta, device=device)
                    dg.save_webp(crop, os.path.join(args.out, crop_path))
                    rec.update(cinfo)
                    rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001 - one bad label must not end the run; the row says what happened
                if dg.is_fatal(e):
                    raise
                import traceback
                traceback.print_exc()
                rec["status"] = f"error:{type(e).__name__}"
            w.writerow(rec)
            f.flush()
            status_counts[rec["status"]] += 1
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} {rec['label_uid']} {rec['status']}; {el / i:.2f} s/label, "
                      f"~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    run = {"labels": args.labels, "labels_sha256": sha256_file(args.labels), "pano_store": os.path.abspath(args.pano_store),
           "pano_store_marker": open(os.path.join(args.pano_store, dg.PANO_STORE_MARKER)).readline().strip(),
           "tilt_beta": args.tilt_beta, "depth_encoder": "vitb", "renderer": "torch" if device else "numpy",
           "package_source": dg.package_source(), "status_counts": dict(status_counts),
           "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(args.out, "crops_run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print(json.dumps(run, indent=2))


def cmd_score(args):
    """``evaluate.run_inference`` on every ok crop of the manifest with one checkpoint -> predictions.csv."""
    import evaluate
    m = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    m = m[m.status == "ok"].copy()
    m["crop_path"] = [os.path.join(os.path.dirname(os.path.abspath(args.manifest)), p) for p in m.crop_path]
    m["split"] = "test"
    pred = evaluate.run_inference(m, args.checkpoint, args.revision)
    pred.to_csv(args.out, index=False)
    weights = os.path.join(args.checkpoint, "model.safetensors")
    prov = {"checkpoint": os.path.abspath(args.checkpoint), "revision": args.revision,
            "model_sha256": sha256_file(weights) if os.path.exists(weights) else None,
            "manifest_sha256": sha256_file(args.manifest), "n": int(len(pred)),
            "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.splitext(args.out)[0] + "_provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def load_run(run_dir):
    labels = pd.read_csv(os.path.join(run_dir, "labels.csv"), dtype={"label_id": str}, low_memory=False)
    manifest = pd.read_csv(os.path.join(run_dir, "manifest.csv"), dtype=str, keep_default_na=False)
    pred = pd.read_csv(os.path.join(run_dir, "predictions.csv"))
    df = labels.merge(manifest[["label_uid", "status", "depth_m", "crop_half_px", "tilt_applied", "pose_unhandled"]],
                      on="label_uid", how="left", validate="one_to_one")
    df = df.merge(pred[["label_uid", "p_correct", "p_incorrect"]], on="label_uid", how="left", validate="one_to_one")
    df["true"] = df.label_class
    df["verdict"] = np.where(df.p_incorrect > df.p_correct, "incorrect", "correct")
    df["confidence"] = np.maximum(df.p_correct, df.p_incorrect)
    df["scored"] = df.p_correct.notna()
    return df


def load_round3(round3_dir):
    """Round-3 human-placed CurbRamp test rows scored by the same checkpoint: the baseline."""
    pred = pd.read_csv(os.path.join(round3_dir, "predictions.csv"))
    split = pd.read_csv(os.path.join(round3_dir, "split.csv"), dtype=str, usecols=["label_uid", "city", "pano_id", "author"])
    df = pred.merge(split, on="label_uid", validate="one_to_one")
    df = df[(df.split == "test") & (df.author == "human")].copy()
    df["verdict"] = np.where(df.p_incorrect > df.p_correct, "incorrect", "correct")
    df["confidence"] = np.maximum(df.p_correct, df.p_incorrect)
    return df


def table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "|".join("---:" if c not in ("slice", "city", "tier", "reference") else "---" for c, _ in cols) + "|"
    body = ["| " + " | ".join(str(f(r)) for _, f in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    df = load_run(args.run)
    api_map = json.load(open(args.map))
    api_map_unc = json.load(open(args.map_uncapped)) if args.map_uncapped else None
    r3 = load_round3(args.round3) if args.round3 else None
    os.makedirs(args.out, exist_ok=True)
    sel = json.load(open(os.path.join(args.run, "selection.json")))
    ai = df[df.author == "ai"]
    scored = ai[ai.scored]
    clean = scored[scored.clean == 1]
    L = [f"<!-- generated by analysis/ai_labels_eval.py report on {datetime.now(timezone.utc).date()} -->", ""]

    # 1. population
    L += ["### Population", "",
          "| city | provider | AI CurbRamp labels | human-voted | panos | Disagree majority | ≥ 2 votes | expert-decided | core / band / mixed | in round-3 train / val / test | cropped | not scored |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for city, g in ai.groupby("city"):
        s = sel["per_city"][city]
        r3c = s["ai_in_round3"]
        L.append(f"| {city} | {'/'.join(s['pano_source'])} | {s['ai_labels_total']:,} | {s['ai_voted']:,} | {s['ai_panos']:,} | "
                 f"{s['ai_incorrect']} ({s['ai_incorrect'] / s['ai_voted']:.1%}) | {s['ai_two_plus_votes']} | {s['ai_expert_decided']:,} | "
                 f"{s['ai_by_tier']['core']} / {s['ai_by_tier']['band']} / {s['ai_by_tier']['mixed']} | "
                 f"{r3c.get('train', 0)} / {r3c.get('val', 0)} / {r3c.get('test', 0)} | {int(g.scored.sum()):,} | "
                 f"{int((~g.scored).sum())} |")
    L.append("")
    not_scored = ai[~ai.scored].status.value_counts().to_dict()
    L.append(f"Not scored (no crop): {not_scored or 'none'}. Ties (equal Agree and Disagree) dropped: {sel['ties_dropped']}.")
    L.append("")

    # 2. headline
    def rows_for(frame, name):
        out = []
        for city, g in [("all three", frame)] + list(frame.groupby("city")):
            b = auc_block(g, args.n_boot)
            b["slice"] = f"{name}, {city}"
            out.append(b)
        return out

    head_rows = rows_for(clean, "unseen") + rows_for(scored, "all scored")
    cols = [("slice", lambda r: r["slice"]), ("n", lambda r: f"{r['n']:,}"), ("incorrect", lambda r: r["n_incorrect"]),
            ("panos", lambda r: f"{r['n_panos']:,}"), ("AUC(incorrect) [95% pano CI]", lambda r: fmt_ci(r["auc"], r["auc_lo"], r["auc_hi"])),
            ("acc", lambda r: fmt(r["acc"])), ("bal. acc", lambda r: fmt(r["bal_acc"])),
            ("recall(incorrect) @ P ≥ 0.92", lambda r: fmt(r["recall_incorrect_at_p92"]))]
    L += ["### Headline: the round-3 CurbRamp validator on AI-placed curb ramps", "", table(head_rows, cols), ""]
    if r3 is not None:
        b = auc_block(r3, args.n_boot)
        b["slice"] = "baseline: human-placed, round-3 test split (all training deployments)"
        L += ["Baseline, same checkpoint, same metrics:", "", table([b], cols), ""]

    # 3. gate
    def gate_rows(frame, name, m, mu):
        served = served_accuracy(m, frame.verdict, frame.confidence)
        g = gate_metrics(frame.true, frame.verdict, served)
        row = {"slice": name, "n": len(frame), "agree": g["agree"], "disagree": g["disagree"], "all": g["all"]}
        if mu is not None:
            served_u = served_accuracy(mu, frame.verdict, frame.confidence)
            row["disagree_unc"] = gate_metrics(frame.true, frame.verdict, served_u)["disagree"]
        return row

    gate_frames = [("unseen, all three", clean)] + [(f"unseen, {c}", g) for c, g in clean.groupby("city")]
    gate_frames += [(f"unseen, tier {t}", clean[clean.tier == t]) for t in TIERS if (clean.tier == t).any()]
    if r3 is not None:
        gate_frames.append(("baseline: human-placed round-3 test", r3))
    grows = [gate_rows(f, n, api_map, api_map_unc) for n, f in gate_frames if len(f)]
    gcols = [("slice", lambda r: r["slice"]), ("n", lambda r: f"{r['n']:,}"),
             ("Agree verdicts", lambda r: r["agree"]["n"]), ("Agree pass", lambda r: r["agree"]["n_pass"]),
             ("Agree coverage", lambda r: fmt(r["agree"]["coverage"])),
             ("Agree precision [Wilson]", lambda r: fmt_ci(r["agree"]["precision"], r["agree"]["precision_lo"], r["agree"]["precision_hi"])),
             ("Disagree verdicts", lambda r: r["disagree"]["n"]), ("Disagree pass (served map)", lambda r: r["disagree"]["n_pass"])]
    if api_map_unc is not None:
        gcols += [("Disagree pass (uncapped)", lambda r: r["disagree_unc"]["n_pass"]),
                  ("Disagree precision (uncapped)", lambda r: fmt_ci(r["disagree_unc"]["precision"], r["disagree_unc"]["precision_lo"], r["disagree_unc"]["precision_hi"])),
                  ("Disagree catch (uncapped)", lambda r: fmt(r["disagree_unc"]["catch"]))]
    L += ["### At the deployed gate (served accuracy ≥ 0.92)", "", table(grows, gcols), ""]

    # 4. slices
    def slice_rows(frame, key, name):
        out = []
        for val, g in frame.groupby(key):
            if len(g) < args.min_n:
                continue
            b = auc_block(g, args.n_boot)
            served = served_accuracy(api_map, g.verdict, g.confidence)
            gm = gate_metrics(g.true, g.verdict, served)
            b.update({"slice": f"{name} = {val}", "human_agree_share": round(float((g.true == "correct").mean()), 4),
                      "agree_cov": gm["agree"]["coverage"], "agree_prec": gm["agree"]["precision"]})
            out.append(b)
        return out

    scols = [("slice", lambda r: r["slice"]), ("n", lambda r: f"{r['n']:,}"), ("incorrect", lambda r: r["n_incorrect"]),
             ("human Agree share", lambda r: fmt(r["human_agree_share"])),
             ("AUC [95% pano CI]", lambda r: fmt_ci(r["auc"], r["auc_lo"], r["auc_hi"])),
             ("recall(incorrect) @ P ≥ 0.92", lambda r: fmt(r["recall_incorrect_at_p92"])),
             ("Agree coverage", lambda r: fmt(r["agree_cov"])), ("Agree precision", lambda r: fmt(r["agree_prec"]))]
    clean2 = clean.copy()
    clean2["city_tier"] = clean2.city + " / " + clean2.tier
    clean2["votes"] = np.where(clean2.n_votes >= 2, "≥ 2", "1")
    clean2["reference"] = np.where(clean2.expert_class != "", "expert-decided", "crowd only")
    clean2["pano_px"] = clean2.pano_width.astype(int).astype(str) + "×" + clean2.pano_height.astype(int).astype(str)
    clean2["vote_dir_expert"] = np.where(clean2.expert_class != "", clean2.expert_class, "")
    srows = (slice_rows(clean2, "city_tier", "city / tier") + slice_rows(clean2, "votes", "votes")
             + slice_rows(clean2, "reference", "reference") + slice_rows(clean2, "pano_px", "pano size"))
    L += ["### Slices (unseen labels)", "", table(srows, scols), ""]
    exp = clean2[clean2.expert_class != ""].copy()
    if len(exp):
        exp["true"] = exp.expert_class
        b = auc_block(exp, args.n_boot)
        L.append(f"Against the expert verdict alone (labels an expert voted on with no contradicting human vote): "
                 f"n {b['n']:,}, incorrect {b['n_incorrect']}, AUC {fmt_ci(b['auc'], b['auc_lo'], b['auc_hi'])}, "
                 f"recall(incorrect) at P ≥ 0.92 {fmt(b['recall_incorrect_at_p92'])}.")
        L.append("")
    sev = ai[(ai.severity.notna()) & (ai.severity.astype(str) != "")]
    tags = ai[ai.tags.astype(str).str.strip().isin(["", "[]"]) == False]  # noqa: E712
    L.append(f"Severity and tags: AI-placed labels carry neither (RampNet submits none); {len(sev)} of {len(ai):,} voted "
             f"labels have a severity and {len(tags)} have tags (edited by a validator afterwards), too few to slice. "
             f"The type × severity and type × tag slices of the standing rule apply to the human-placed baseline in "
             f"`docs/retrain-2026-10.md`, not here.")
    L.append("")

    # 5. human-placed labels in the same deployments
    hum = df[(df.author == "human") & df.scored]
    if len(hum):
        hrows = []
        for city, g in hum.groupby("city"):
            b = auc_block(g, args.n_boot)
            b["slice"] = f"human-placed, {city}" + (" (pose unhandled: untilted)" if (g.pose_unhandled.astype(str) == "1").any() else "")
            hrows.append(b)
        L += ["### Human-placed CurbRamp labels in the same deployments (tiny)", "", table(hrows, cols), ""]

    # 6. confidence distribution
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    qrows = [("unseen AI-placed, Agree verdicts", clean[clean.verdict == "correct"].confidence),
             ("unseen AI-placed, Disagree verdicts", clean[clean.verdict == "incorrect"].confidence)]
    if r3 is not None:
        qrows += [("baseline Agree verdicts", r3[r3.verdict == "correct"].confidence),
                  ("baseline Disagree verdicts", r3[r3.verdict == "incorrect"].confidence)]
    L += ["### Confidence of the verdicts", "", "| slice | n | " + " | ".join(f"p{int(x * 100)}" for x in q) + " |",
          "|---|---:|" + "---:|" * len(q)]
    for name, s in qrows:
        L.append(f"| {name} | {len(s):,} | " + " | ".join(f"{v:.4f}" for v in s.quantile(q)) + " |")
    L.append("")

    # 7. provenance
    crops_run = json.load(open(os.path.join(args.run, "crops_run.json")))
    prov = json.load(open(os.path.join(args.run, "predictions_provenance.json")))
    L += ["### Provenance", "",
          f"- rawLabels pulls: " + "; ".join(f"{c} `{v['sha256'][:12]}`" for c, v in sel["inputs"].items()) + f" (selected {sel['selected_on']})",
          f"- labels.csv `{sel['labels_csv_sha256'][:12]}`; {sel['n_labels']:,} labels on {sel['n_panos']:,} panos; round-3 split `{(sel['round3_split'] or {}).get('sha256', '')[:12]}`",
          f"- crops: store marker \"{crops_run['pano_store_marker']}\", {crops_run['package_source']}, tilt beta {crops_run['tilt_beta']}, depth {crops_run['depth_encoder']}, status {crops_run['status_counts']}",
          f"- checkpoint `{os.path.basename(prov['checkpoint'])}` model.safetensors `{(prov['model_sha256'] or '')[:12]}`, {prov['n']:,} crops scored {prov['scored']}",
          f"- accuracy map `{os.path.relpath(args.map)}` `{sha256_file(args.map)[:12]}`"
          + (f"; uncapped `{sha256_file(args.map_uncapped)[:12]}`" if args.map_uncapped else ""),
          f"- bootstrap: {args.n_boot} panorama resamples, seed {SEED}; slices need ≥ {args.min_n} rows", ""]
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write("\n".join(L))

    # figures: ROC per city + confidence histograms
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for name, g in [("baseline (human-placed, round-3 test)", r3)] + [(f"AI-placed, {c}", g) for c, g in clean.groupby("city")]:
        if g is None or g.true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve((g.true == "incorrect").astype(int), g.p_incorrect)
        ax.plot(fpr, tpr, label=f"{name} (n={len(g):,})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false positive rate (correct labels called incorrect)")
    ax.set_ylabel("true positive rate (incorrect labels caught)")
    ax.set_title("ROC, round-3 CurbRamp validator")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    bins = np.linspace(0.5, 1.0, 26)
    for name, g in [("baseline", r3)] + [(f"AI-placed, {c}", g) for c, g in clean.groupby("city")]:
        if g is None:
            continue
        ax.hist(g.confidence, bins=bins, histtype="step", density=True, label=name)
    ax.set_xlabel("verdict confidence (max softmax)")
    ax.set_ylabel("density")
    ax.set_title("Confidence distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "roc_confidence.png"), dpi=130)
    print("\n".join(L))


# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="label set, pano list and provenance from rawLabels pulls")
    s.add_argument("--rawlabels", required=True, help="directory with <city_id>.csv rawLabels exports")
    s.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    s.add_argument("--round3-split", default=None, help="round-3 CurbRamp split.csv, to flag labels it trained on")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_select)
    c = sub.add_parser("crops", help="cut the crops from a staged pano store copy (GPU)")
    c.add_argument("--labels", required=True)
    c.add_argument("--pano-store", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cities", default=None, help="comma-separated subset")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--tilt-beta", type=float, default=dg.DEFAULT_TILT_BETA)
    c.add_argument("--allow-cpu", action="store_true")
    c.set_defaults(func=cmd_crops)
    sc = sub.add_parser("score", help="predictions.csv from the manifest's ok crops with one checkpoint (GPU)")
    sc.add_argument("--manifest", required=True)
    sc.add_argument("--checkpoint", required=True)
    sc.add_argument("--revision", default=None)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=cmd_score)
    r = sub.add_parser("report", help="tables.md and figures from a run directory")
    r.add_argument("--run", required=True, help="directory with labels.csv, selection.json, manifest.csv, predictions.csv")
    r.add_argument("--map", required=True, help="the served CurbRamp accuracy map (capped, as deployed)")
    r.add_argument("--map-uncapped", default=None, help="the uncapped map, for what Disagree would do if not held back")
    r.add_argument("--round3", default=None, help="round-3 CurbRamp dir with predictions.csv + split.csv (baseline)")
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--min-n", type=int, default=30)
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
