"""The pre-registered Disagree bar of the round-4 recipe (issue #74): fit each candidate's uncapped maps, judge them on
test, on unseen deployments and on the expert gold set, and apply the decision rules of ``docs/round4-2026-11.md``.

Why
---
Production holds back every Disagree map (the served maps are capped just under 0.92). The bar below says, before
any model is scored, what a Disagree map must show to become ``shadow_eligible``: a published side file replayed
offline over the confidences production records, with no vote and no user-visible change. Nothing here can emit a
status above ``shadow_eligible``; a Disagree vote in a deployment (Tier B) needs production evidence and an expert
spot check that this module does not produce.

Subcommands:

- ``maps`` (CPU): per (type, run, seed), the uncapped map fitted on ``val`` under the ``majority2`` prior
  (``prior_rules.val_map``), gate statistics for both verdicts on ``test`` and ``test_heldout`` (``gate_stats``),
  the same under the ``margin`` prior as a reference row, the map's maximum Disagree value, a fit on ``test``
  evaluated on ``test_heldout`` (fit-split sensitivity), test / held-out AUC, and the paired held-out contrast of
  ``cl-drop`` against the gold-free baseline (mean of per-seed deltas, one panorama resample shared by all seeds).
- ``gold`` (Mac only; the only place expert verdicts are read): joins the gold predictions to ``gold.csv`` and
  computes the gold rows of the bar (gated Disagrees, false alarms on hard-correct items, round-stratified AUC, the
  primary paired contrast P4, strata, rounds, slices, sensitivity rows, crop parity).
- ``mde``: the minimum detectable AUC difference per type for P4 from verdict counts only (Hanley and McNeil 1982,
  doi:10.1148/radiology.143.1.7063747; paired correlation an assumption, stated).
- ``decide``: applies the rules mechanically; one record per type with each criterion's number and pass / fail /
  unestimable and a status in ``{hold_back, shadow_eligible}``.
- ``report``: splices the generated blocks into the doc after a privacy check against the live ``/v3/api/cities``.

Every output is aggregate. Per-item files are refused inside a git checkout (``adjudicator.refuse_in_checkout``).

Usage::

    python analysis/disagree_gate.py mde --gold ~/Data/.../gold.csv --out docs/round4-2026-11
    python analysis/disagree_gate.py maps --r4 $R4 --shares docs/deployment-prior-2026-11/shares.json \\
        --runs baseline,cl-drop,round3 --out ~/Data/sidewalk-validator-ai-round4-2026-11/gate
    python analysis/disagree_gate.py gold --gold ~/Data/.../gold.csv --pred-dir $PRED --maps $GATE/maps \\
        --pool ~/Data/.../relabel_pool.csv --crops uniform --out $GATE
    python analysis/disagree_gate.py decide --maps $GATE/maps.json --gold $GATE/gold.json --out $GATE/bar.json
    python analysis/disagree_gate.py report --dir $GATE --doc docs/round4-2026-11.md
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from analysis import prior_rules as pr  # noqa: E402
from analysis import slices as sl  # noqa: E402
from calibration import _isotonic_map as im  # noqa: E402
from calibration import calibrate as cal  # noqa: E402

LABEL_TYPES = ("Crosswalk", "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem")
CL_DROP_TYPES = ("Crosswalk", "CurbRamp", "NoCurbRamp", "Obstacle")
GATE = im.GATE
Z95 = 1.96
VERDICTS = ("correct", "incorrect")          # the API's Agree and Disagree
VERDICT_NAME = {"correct": "Agree", "incorrect": "Disagree"}
EVAL_SPLITS = ("test", "test_heldout")
PRIOR_RULE, REFERENCE_RULE = "majority2", "margin"
SEEDS = (42, 43, 44)
SHIP_SEED = 42
N_BOOT = 2000
# The bar (docs/round4-2026-11.md, "Decision rules"). Judgment calls, listed for Jon under J2.
A1_MIN_N = 30            # gated Disagrees on test
A2_MIN_N = 10            # gated Disagrees on unseen deployments for A2 to be estimable
A3_MIN_N = 20            # gated Disagrees on gold for the share rule; 1..19 is thin (none may be overturned)
A3_SHARE, A3_LOWER = 0.92, 0.80
C2_MAX_AUC_LOSS = 0.010  # cl-drop's mean test AUC may not fall further below the baseline's
C3_MAX_COVERAGE_LOSS = 0.03   # Agree coverage at the gate, absolute
C3_MAX_PRECISION_LOSS = 0.01  # prior-weighted Agree precision, relative to the gold-free baseline (A6)
PRIVACY_MIN_CELL = 5     # a gold cell under 5 items shows its size only
ROUND_FLAG_RATIO = 2.0   # A3 per round: flag a round overturned at twice the other's rate
STATUSES = ("hold_back", "shadow_eligible")
BEGIN = "<!-- BEGIN GENERATED: {name} (analysis/disagree_gate.py) -->"
END = "<!-- END GENERATED: {name} -->"
R3CROPS_CAPTION = ("Round-3 crops only; crop source and verdict are confounded; not P4, not A3 (descriptive, and the "
                   "input of the crop-parity table).")


# ---------------------------------------------------------------------------------------------------------------------
# Gate statistics
# ---------------------------------------------------------------------------------------------------------------------

def wilson(k, n, z=Z95):
    """95% Wilson interval of ``k`` of ``n``; NaNs for ``n == 0``.

    >>> [round(x, 3) for x in wilson(19, 20)]
    [0.764, 0.991]
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def reweight_precision(p, ratio):
    """Precision ``p`` re-expressed under prior weights that give a *wrong* row ``ratio`` times the weight of a right
    one: ``p / (p + ratio (1 - p))`` (``nosidewalk_gate.reweight_precision``, here for either verdict). Monotone, so
    it maps a Wilson bound on the raw precision to an exact bound on the weighted one.

    >>> round(reweight_precision(0.9, 0.25), 4), reweight_precision(1.0, 3.0)
    (0.973, 1.0)
    """
    p = float(p)
    if np.isnan(p) or np.isnan(ratio):
        return float("nan")
    return p / (p + ratio * (1.0 - p)) if 0.0 < p < 1.0 else p


def raw_precision_needed(ratio, gate=GATE):
    """The raw precision at which the prior-weighted one equals ``gate``: solving ``p / (p + r (1 - p)) = g`` gives
    ``p = g r / (1 - g + g r)``.

    >>> round(raw_precision_needed(1.0), 4), round(reweight_precision(raw_precision_needed(4.0), 4.0), 4)
    (0.92, 0.92)
    """
    if np.isnan(ratio):
        return float("nan")
    return gate * ratio / (1 - gate + gate * ratio)


def gate_stats(df, verdict, pi, gate=GATE):
    """Gate statistics of one verdict on rows carrying ``verdict``, ``served`` and ``true``.

    ``n_pass`` gated rows; ``coverage`` (raw) and ``coverage_weighted`` (prior-weighted share of all rows) gated;
    raw precision (the gated rows whose true class is the verdict) with its Wilson 95% interval; the prior-weighted
    precision (``calibrate.prior_weights`` under ``pi``) and its exact lower bound (the raw bound mapped through the
    class weight ratio); ``raw_precision_needed`` for the weighted precision to reach ``gate``. Weighted fields are
    NaN when a class is absent from ``df``.

    >>> d = pd.DataFrame({"verdict": ["incorrect"] * 3 + ["correct"], "served": [0.95, 0.99, 0.5, 0.97],
    ...                   "true": ["incorrect", "correct", "incorrect", "correct"]})
    >>> g = gate_stats(d, "incorrect", {"correct": 0.9, "incorrect": 0.1})
    >>> g["n_pass"], g["precision"], round(g["weight_ratio"], 3), round(g["precision_weighted"], 3)
    (2, 0.5, 9.0, 0.1)
    """
    n = len(df)
    true = df["true"].astype(str).to_numpy()
    passed = ((df["verdict"].to_numpy() == verdict) & (df["served"].to_numpy() >= gate))
    right = (true == verdict).astype(float)
    k = int(right[passed].sum())
    n_pass = int(passed.sum())
    lo, hi = wilson(k, n_pass)
    out = {"verdict": VERDICT_NAME[verdict], "n": int(n), "n_verdict": int((df["verdict"] == verdict).sum()),
           "n_pass": n_pass, "n_right": k, "n_wrong": n_pass - k,
           "coverage": n_pass / n if n else float("nan"),
           "precision": k / n_pass if n_pass else float("nan"), "precision_lo": lo, "precision_hi": hi,
           "coverage_weighted": float("nan"), "precision_weighted": float("nan"), "weight_ratio": float("nan"),
           "precision_weighted_lo": float("nan"), "raw_precision_needed": float("nan")}
    try:
        w = cal.prior_weights(df["true"], pi)
    except ValueError:
        return out
    if not ((true == "correct").any() and (true == "incorrect").any()):
        return out
    other = "correct" if verdict == "incorrect" else "incorrect"
    ratio = float(w[true == other][0] / w[true == verdict][0])
    out["weight_ratio"] = ratio
    out["coverage_weighted"] = float(w[passed].sum() / w.sum())
    out["raw_precision_needed"] = raw_precision_needed(ratio, gate)
    if n_pass:
        out["precision_weighted"] = float(np.average(right[passed], weights=w[passed]))
        out["precision_weighted_lo"] = reweight_precision(lo, ratio)
    return out


def served_with_map(df, api_map):
    """``verdict``, ``confidence`` and ``served`` from ``api_map`` for rows with ``p_correct`` / ``p_incorrect``."""
    d = cal.verdicts(df)
    served = np.zeros(len(d))
    for v in VERDICTS:
        m = (d["verdict"] == v).to_numpy()
        served[m] = im.api_lookup_many(api_map[v], d["confidence"].to_numpy()[m])
    return d.assign(served=served)


def fit_map(pred, pi, fit_split="val"):
    """The uncapped map fitted on ``fit_split`` under prior ``pi`` (``prior_rules.val_map`` for ``val``)."""
    d = cal.verdicts(pred)
    curves, _ = cal.fit_curves(d[d["split"] == fit_split], pi)
    amap = im.build_api_map(curves)
    im.validate_api_map(amap)
    return amap


# ---------------------------------------------------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------------------------------------------------

def run_tag(variant, seed):
    return variant if seed == SHIP_SEED or variant == "round3" else f"{variant}-s{seed}"


def load_shares_prior(shares_path, label_type, rule):
    with open(shares_path) as f:
        shares = json.load(f)
    return pr.rule_prior(shares["types"], label_type, rule)


def check_prior_equal(shares_pi, priors_path, label_type, rule=PRIOR_RULE, tol=1e-12):
    """Raise unless the ``rule`` prior of a ``deployment_priors.json`` with a ``rules`` block equals the one from
    ``shares.json`` (the two sources must not drift)."""
    pi, _ = cal.load_prior(priors_path, label_type, rule=rule)
    if any(abs(pi[c] - shares_pi[c]) > tol for c in VERDICTS):
        raise ValueError(f"{label_type}: {rule} prior differs between shares.json {shares_pi} and {priors_path} {pi}")


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def split_auc(pred, split):
    p = pred[pred["split"] == split]
    return auc((p["true"] == "incorrect").astype(int).to_numpy(), p["p_incorrect"].to_numpy())


def map_record(pred, pi, pi_ref):
    """One run's map statistics (see ``maps``)."""
    amap = fit_map(pred, pi)
    scored = served_with_map(pred, amap)
    rec = {"map_max": {VERDICT_NAME[v]: float(max(amap[v].values())) for v in VERDICTS},
           "gates": {}, "reference_margin": {}, "fit_on_test_eval_heldout": {},
           "auc": {s: split_auc(pred, s) for s in EVAL_SPLITS}}
    for s in EVAL_SPLITS:
        e = scored[scored["split"] == s]
        if len(e):
            rec["gates"][s] = {VERDICT_NAME[v]: gate_stats(e, v, pi) for v in VERDICTS}
    ref = served_with_map(pred, fit_map(pred, pi_ref))
    for s in EVAL_SPLITS:
        e = ref[ref["split"] == s]
        if len(e):
            rec["reference_margin"][s] = {VERDICT_NAME[v]: gate_stats(e, v, pi_ref) for v in VERDICTS}
    if (pred["split"] == "test").any() and (pred["split"] == "test_heldout").any():
        t = served_with_map(pred, fit_map(pred, pi, "test"))
        e = t[t["split"] == "test_heldout"]
        rec["fit_on_test_eval_heldout"] = {VERDICT_NAME[v]: gate_stats(e, v, pi) for v in VERDICTS}
    return rec, amap


def paired_seed_delta(preds_a, preds_b, groups, split, n_boot=N_BOOT, seed=0):
    """The mean over seeds of the paired AUC difference ``b - a`` on ``split`` (``preds_a`` / ``preds_b``:
    ``{seed: predictions}``, the same seeds), with a 95% percentile interval from resampling whole panoramas once
    per replicate and applying the same resample to every seed and both arms. Per-seed deltas are returned too."""
    seeds = sorted(set(preds_a) & set(preds_b))
    if not seeds:
        return None
    frames = {}
    for s in seeds:
        a = preds_a[s][preds_a[s]["split"] == split][["label_uid", "true", "p_incorrect"]]
        b = preds_b[s][preds_b[s]["split"] == split][["label_uid", "p_incorrect"]]
        frames[s] = a.merge(b, on="label_uid", suffixes=("_a", "_b"), validate="one_to_one")
    base = frames[seeds[0]][["label_uid", "true"]]
    for s in seeds[1:]:
        if not frames[s]["label_uid"].equals(base["label_uid"]):
            frames[s] = base[["label_uid"]].merge(frames[s], on="label_uid", validate="one_to_one")
    y = (base["true"] == "incorrect").astype(int).to_numpy()
    g = base["label_uid"].map(groups).fillna(base["label_uid"]).to_numpy()
    per = {s: auc(y, frames[s]["p_incorrect_b"]) - auc(y, frames[s]["p_incorrect_a"]) for s in seeds}
    out = {"split": split, "seeds": seeds, "per_seed": per, "mean": float(np.mean(list(per.values()))),
           "lo": float("nan"), "hi": float("nan"), "n": int(len(y))}
    if n_boot:
        rng = np.random.default_rng(seed)
        idxs = sl.boot_indices(y, g, n_boot, rng)
        boots = []
        for idx in idxs:
            yy = y[idx]
            boots.append(np.mean([auc(yy, frames[s]["p_incorrect_b"].to_numpy()[idx])
                                  - auc(yy, frames[s]["p_incorrect_a"].to_numpy()[idx]) for s in seeds]))
        boots = np.asarray(boots, dtype=float)
        boots = boots[~np.isnan(boots)]
        if len(boots):
            out["lo"], out["hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return out


def load_run_predictions(r4, label_type, tag):
    p = os.path.join(r4, label_type, f"predictions_{tag}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, dtype={"label_uid": str, "split": str, "true": str})


def cmd_maps(args):
    types = [t for t in args.types.split(",") if t]
    runs = [r for r in args.runs.split(",") if r]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"_prior_rule": args.prior_rule, "_reference_rule": REFERENCE_RULE, "_gate": GATE, "_seeds": seeds}
    os.makedirs(os.path.join(args.out, "maps"), exist_ok=True)
    for t in types:
        pi = load_shares_prior(args.shares, t, args.prior_rule)
        pi_ref = load_shares_prior(args.shares, t, REFERENCE_RULE)
        if args.priors:
            check_prior_equal(pi, args.priors, t, args.prior_rule)
        rec = {"prior": pi, "prior_reference": pi_ref, "runs": {}}
        preds = {}
        for variant in runs:
            if variant == "cl-drop" and t not in CL_DROP_TYPES:
                continue
            for s in ([SHIP_SEED] if variant == "round3" else seeds):
                tag = run_tag(variant, s)
                pred = load_run_predictions(args.r4, t, tag)
                if pred is None:
                    rec["runs"][tag] = {"variant": variant, "seed": s, "missing": True}
                    continue
                preds.setdefault(variant, {})[s] = pred
                r, amap = map_record(pred, pi, pi_ref)
                rec["runs"][tag] = {"variant": variant, "seed": s, **r}
                with open(os.path.join(args.out, "maps", f"{t}_{tag}.json"), "w") as f:
                    json.dump(amap, f, indent=1)
        split_path = os.path.join(args.r4, t, "split.csv")
        if "baseline" in preds and "cl-drop" in preds and os.path.exists(split_path):
            sp = pd.read_csv(split_path, dtype=str, keep_default_na=False)
            groups = dict(zip(sp["label_uid"], sp["city"] + "|" + sp["pano_id"]))
            rec["contrasts"] = {"cl-drop": {s: paired_seed_delta(preds["baseline"], preds["cl-drop"], groups, s,
                                                                 args.n_boot, args.seed) for s in EVAL_SPLITS}}
        out[t] = rec
        print(f"{t}: {sum(1 for r in rec['runs'].values() if not r.get('missing'))} runs with predictions")
    path = os.path.join(args.out, "maps.json")
    with open(path, "w") as f:
        json.dump(cal.without_nan(out), f, indent=2, default=float)
        f.write("\n")
    print(f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# gold (Mac only)
# ---------------------------------------------------------------------------------------------------------------------

def _truthy(col):
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "t"])


def gold_population(gold, val_back=False, consensus_back=False):
    """Masks over ``gold``: ``decided`` (expert correct or incorrect, the item decided, masked) and ``headline``
    (decided, not a round-2 consensus verdict, not a round-3 val row). The sensitivity rows put the val items or
    the consensus verdicts back. Undecided and unmasked items are never counted."""
    decided = (_truthy(gold["decided"]) & gold["expert"].isin(VERDICTS) & ~_truthy(gold["unmasked"]))
    head = decided.copy()
    if not consensus_back:
        head &= gold["expert_source"] != "r2_consensus"
    if not val_back:
        head &= gold["round3_split"] != "val"
    return decided, head


def stratified_auc(y, s, strata):
    """The round-stratified AUC: per-stratum AUCs averaged with weights n_correct x n_incorrect (the weighting of
    ``retrain_report.within_city_auc``, with the round as the stratum); strata with one class are skipped.

    >>> y = np.array([0, 1, 0, 1, 0, 0, 1, 1]); s = np.array([.1, .9, .2, .3, .5, .6, .4, .7])
    >>> round(stratified_auc(y, s, np.array([1, 1, 1, 1, 2, 2, 2, 2])), 4)
    0.75
    """
    y, s, strata = np.asarray(y), np.asarray(s, float), np.asarray(strata)
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        n1, n0 = int(y[m].sum()), int(m.sum() - y[m].sum())
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def pool_weights(gold_rows, pool):
    """Weight per gold item = pool stratum count / gold stratum count, per (label type, stratum) (``pool`` already
    carries ``label_type`` and ``stratum``). None when the pool is missing or a stratum has no pool count."""
    if pool is None:
        return None
    pc = pool.groupby(["label_type", "stratum"]).size()
    gc = gold_rows.groupby(["label_type", "stratum"]).size()
    keys = list(zip(gold_rows["label_type"], gold_rows["stratum"]))
    if any(k not in pc.index for k in keys):
        return None
    return np.array([pc[k] / gc[k] for k in keys], dtype=float)


def gated_block(rows, verdict, weights=None):
    """Gated items of ``verdict`` among ``rows`` (``verdict``, ``served``, ``expert``): n, expert-confirmed share
    with Wilson interval, overturned count, and the pool-weighted share when ``weights`` is given."""
    m = (rows["verdict"] == verdict).to_numpy() & (rows["served"].to_numpy() >= GATE)
    ok = (rows["expert"].to_numpy() == verdict)
    n, k = int(m.sum()), int((m & ok).sum())
    lo, hi = wilson(k, n)
    out = {"n": n, "confirmed": k, "overturned": n - k, "share": k / n if n else float("nan"), "lo": lo, "hi": hi,
           "share_pool_weighted": float("nan"), "weighting": "raw only" if weights is None else "pool stratum"}
    if weights is not None and n:
        out["share_pool_weighted"] = float(np.average(ok[m], weights=weights[m]))
    return out


def gold_rows_for_run(gold, pred, amap):
    """``gold`` joined one to one with one run's predictions (only the items it scored), with verdict and served."""
    m = gold.merge(pred[["label_uid", "p_correct", "p_incorrect"]], on="label_uid", how="inner",
                   validate="one_to_one")
    return served_with_map(m, amap) if len(m) else m.assign(verdict=[], served=[], confidence=[], correct=[])


def gold_run_stats(rows, weights=None):
    """Per run: gated Disagrees and Agrees (A3 and the check row), hard-correct false alarms, AUCs, by round."""
    ai_dis = rows["ai_vote"] == "Disagree"
    hard_correct = rows[ai_dis & (rows["expert"] == "correct")]
    y = (rows.loc[ai_dis, "expert"] == "incorrect").astype(int).to_numpy()
    s = rows.loc[ai_dis, "p_incorrect"].to_numpy(dtype=float)
    rounds = rows.loc[ai_dis, "round"].to_numpy()
    out = {"n_scored": int(len(rows)), "n_ai_disagree": int(ai_dis.sum()),
           "gated_disagree": gated_block(rows, "incorrect", weights),
           "gated_agree": gated_block(rows, "correct", weights),
           "false_alarm_gated": float(((hard_correct["verdict"] == "incorrect")
                                       & (hard_correct["served"] >= GATE)).mean()) if len(hard_correct) else float("nan"),
           "false_alarm_any": float((hard_correct["verdict"] == "incorrect").mean()) if len(hard_correct)
           else float("nan"), "n_hard_correct": int(len(hard_correct)),
           "auc_round_stratified": stratified_auc(y, s, rounds), "auc_pooled": auc(y, s),
           "by_round": {}}
    for r in sorted(set(rows["round"])):
        rr = rows[rows["round"] == r]
        m = rr["ai_vote"] == "Disagree"
        out["by_round"][str(r)] = {"gated_disagree": gated_block(rr, "incorrect"),
                                   "auc": auc((rr.loc[m, "expert"] == "incorrect").astype(int).to_numpy(),
                                              rr.loc[m, "p_incorrect"].to_numpy(dtype=float))}
    rates = [v["gated_disagree"]["overturned"] / v["gated_disagree"]["n"] for v in out["by_round"].values()
             if v["gated_disagree"]["n"]]
    # A3 per round: a round whose gated Disagrees are overturned at twice the other's rate is flagged.
    out["round_flag"] = bool(len(rates) == 2 and max(rates) > 0 and max(rates) >= ROUND_FLAG_RATIO * min(rates))
    out["slices"] = gold_slices(rows)
    return out


def gold_slices(rows):
    """Type x severity and type x tag on the scored gold items: items, gated Disagrees and how many the expert
    confirmed (the training count of each slice per variant comes from ``round4.py slice-train``)."""
    out = {}
    for by in sl.BY:
        d = sl.with_slice(rows, by)
        gated = (d["verdict"] == "incorrect") & (d["served"] >= GATE)
        out[by] = {str(k): {"n": int(len(g)), "gated_disagree": int(gated[g.index].sum()),
                            "confirmed": int((gated[g.index] & (g["expert"] == "incorrect")).sum())}
                   for k, g in d.groupby("slice")}
    return out


def paired_gold_delta(rows_a, rows_b, n_boot=N_BOOT, seed=0):
    """P4: per seed, the round-stratified AUC of ``b`` minus ``a`` on the same AI-Disagree items (``rows_a`` /
    ``rows_b``: ``{seed: joined rows}``); the primary is the **mean of the per-seed deltas**, with a 95% percentile
    interval from resampling items (one resample shared by every seed and both arms)."""
    seeds = sorted(set(rows_a) & set(rows_b))
    if not seeds:
        return None
    ref = rows_a[seeds[0]]
    ref = ref[ref["ai_vote"] == "Disagree"][["label_uid", "expert", "round"]]
    for s in seeds:
        for rows in (rows_a[s], rows_b[s]):
            ref = ref[ref["label_uid"].isin(rows["label_uid"])]
    y = (ref["expert"] == "incorrect").astype(int).to_numpy()
    rounds = ref["round"].to_numpy()

    def scores(rows):
        return ref[["label_uid"]].merge(rows[["label_uid", "p_incorrect"]], on="label_uid",
                                        validate="one_to_one")["p_incorrect"].to_numpy(dtype=float)

    sa = {s: scores(rows_a[s]) for s in seeds}
    sb = {s: scores(rows_b[s]) for s in seeds}
    per = {s: stratified_auc(y, sb[s], rounds) - stratified_auc(y, sa[s], rounds) for s in seeds}
    out = {"seeds": seeds, "per_seed": per, "mean": float(np.mean(list(per.values()))), "n": int(len(y)),
           "lo": float("nan"), "hi": float("nan")}
    if n_boot and len(y):
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boots.append(np.mean([stratified_auc(y[idx], sb[s][idx], rounds[idx])
                                  - stratified_auc(y[idx], sa[s][idx], rounds[idx]) for s in seeds]))
        boots = np.asarray(boots, float)
        boots = boots[~np.isnan(boots)]
        if len(boots):
            out["lo"], out["hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return out


def mask_thin(n, k=None, min_cell=PRIVACY_MIN_CELL):
    """A count cell for the published tables: under ``min_cell`` items, only the size is shown (``"n=3"``), and a
    share is never printed.

    >>> mask_thin(3, 2), mask_thin(12, 11)
    ('n=3', '11/12')
    """
    if n < min_cell:
        return f"n={n}"
    return f"{k}/{n}" if k is not None else str(n)


def crop_parity(r3, uni, amap):
    """Items scored with both crops: mean |delta p_incorrect|, verdict flips at 0.5 and at the gate."""
    m = r3[["label_uid", "p_correct", "p_incorrect"]].merge(uni[["label_uid", "p_correct", "p_incorrect"]],
                                                           on="label_uid", suffixes=("_r3", "_u"),
                                                           validate="one_to_one")
    if not len(m):
        return {"n": 0}
    a = served_with_map(m.rename(columns={"p_correct_r3": "p_correct", "p_incorrect_r3": "p_incorrect"}), amap)
    b = served_with_map(m.rename(columns={"p_correct_u": "p_correct", "p_incorrect_u": "p_incorrect"}), amap)
    gate_a = np.where(a["served"] >= GATE, a["verdict"], "none")
    gate_b = np.where(b["served"] >= GATE, b["verdict"], "none")
    return {"n": int(len(m)), "mean_abs_dp": float(np.mean(np.abs(m["p_incorrect_r3"] - m["p_incorrect_u"]))),
            "flip_05": float(np.mean(a["verdict"].to_numpy() != b["verdict"].to_numpy())),
            "flip_gate": float(np.mean(gate_a != gate_b))}


def load_pool(path):
    if not path or not os.path.exists(path):
        return None
    from analysis import gt_rounds as gr
    p = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("h_agree", "h_dis", "h_unsure"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0).astype(int)
    try:
        p["stratum"] = gr.assign_stratum(p)
    except (KeyError, ValueError):
        return None
    return p[p["stratum"] != ""]


def cmd_gold(args):
    from analysis import adjudicator as adj
    adj.refuse_in_checkout(args.out, "the gold results (they are built from expert verdicts)")
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    if not gold["label_uid"].is_unique:
        raise SystemExit("gold.csv: label_uid is not unique")
    pool = load_pool(args.pool)
    types = [t for t in args.types.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [r for r in args.runs.split(",") if r]
    res = {"_crops": args.crops, "_caption": R3CROPS_CAPTION if args.crops == "r3crops" else None}
    for t in types:
        g = gold[gold["label_type"] == t]
        _, head = gold_population(g)
        populations = {"headline": g[head], "val_back": g[gold_population(g, val_back=True)[1]],
                       "consensus_back": g[gold_population(g, consensus_back=True)[1]]}
        tr = {"populations": {k: int(len(v)) for k, v in populations.items()}, "runs": {}, "sensitivity": {}}
        joined = {}
        for variant in runs:
            if variant == "cl-drop" and t not in CL_DROP_TYPES:
                continue
            for s in ([SHIP_SEED] if variant == "round3" else seeds):
                tag = run_tag(variant, s)
                pp = os.path.join(args.pred_dir, t, f"gold_predictions_{args.crops}_{tag}.csv")
                mp = os.path.join(args.maps, f"{t}_{tag}.json")
                if not (os.path.exists(pp) and os.path.exists(mp)):
                    tr["runs"][tag] = {"missing": True}
                    continue
                pred = pd.read_csv(pp, dtype={"label_uid": str})
                with open(mp) as f:
                    amap = json.load(f)
                rows = gold_rows_for_run(populations["headline"], pred, amap)
                joined.setdefault(variant, {})[s] = rows
                w = pool_weights(rows, pool)
                tr["runs"][tag] = {"variant": variant, "seed": s, **gold_run_stats(rows, w)}
                if variant == "round3":  # approximation of "scenes it did not train on": not a round-3 train row
                    nr = rows[rows["round3_split"] != "train"]
                    tr["sensitivity"]["round3_not_train_rows"] = gold_run_stats(nr)
                if s == SHIP_SEED:
                    tr["sensitivity"][tag] = {k: gold_run_stats(gold_rows_for_run(populations[k], pred, amap))
                                              for k in ("val_back", "consensus_back")}
                if args.parity and args.crops == "uniform":
                    rp = os.path.join(args.pred_dir, t, f"gold_predictions_r3crops_{tag}.csv")
                    if os.path.exists(rp):
                        tr["runs"][tag]["parity"] = crop_parity(pd.read_csv(rp, dtype={"label_uid": str}), pred, amap)
        if "baseline" in joined and "cl-drop" in joined and args.crops == "uniform":
            tr["p4"] = paired_gold_delta(joined["baseline"], joined["cl-drop"], args.n_boot, args.seed)
        res[t] = tr
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "gold.json")
    with open(path, "w") as f:
        json.dump(cal.without_nan(res), f, indent=2, default=float)
        f.write("\n")
    print(f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# mde
# ---------------------------------------------------------------------------------------------------------------------

def hanley_mcneil_se(a, n_pos, n_neg):
    """Standard error of an AUC ``a`` with ``n_pos`` positives and ``n_neg`` negatives (Hanley and McNeil 1982).

    >>> round(hanley_mcneil_se(0.75, 60, 40), 4)
    0.048
    """
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))


def mde(n_pos, n_neg, auc_assumed=0.75, r=0.8, alpha=0.05, power=0.8):
    """Minimum detectable paired AUC difference (two-sided ``alpha``, ``power``) for two models scored on the same
    items, both at ``auc_assumed``, with correlation ``r`` between their AUC estimates:
    ``(z_{1-a/2} + z_{power}) * sqrt(2 SE^2 (1 - r))``.

    >>> round(mde(60, 40), 3)
    0.085
    """
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    se = hanley_mcneil_se(auc_assumed, n_pos, n_neg)
    return z * math.sqrt(2 * se * se * (1 - r))


def mde_table(gold, types=LABEL_TYPES, auc_assumed=0.75, r=0.8):
    """Per type, on P4's population (decided, masked-only, AI-Disagree, not a round-3 val row; verdict counts only):
    n incorrect / correct and the minimum detectable difference."""
    _, head = gold_population(gold)
    rows = []
    for t in types:
        d = gold[head & (gold["label_type"] == t) & (gold["ai_vote"] == "Disagree")]
        n_pos, n_neg = int((d["expert"] == "incorrect").sum()), int((d["expert"] == "correct").sum())
        rows.append({"label_type": t, "n_expert_incorrect": n_pos, "n_expert_correct": n_neg,
                     "mde": mde(n_pos, n_neg, auc_assumed, r) if n_pos > 1 and n_neg > 1 else float("nan")})
    return rows


def mde_markdown(rows, auc_assumed=0.75, r=0.8):
    lines = ["| Type | expert incorrect | expert correct | minimum detectable AUC difference |",
             "|---|---:|---:|---:|"]
    for x in rows:
        lines.append(f"| {x['label_type']} | {x['n_expert_incorrect']} | {x['n_expert_correct']} | "
                     f"{x['mde']:.3f} |")
    lines += ["", f"Two-sided 5% test at 80% power, Hanley-McNeil standard error at an assumed AUC of "
              f"{auc_assumed:.2f}, paired correlation {r:.1f} between the two models' AUC estimates (an assumption; "
              "the bootstrap half-width replaces it once predictions exist). Population: decided, masked-only gold "
              "items whose 2025-08 AI vote was Disagree, not a round-3 val row; the uniform-crop filter of stage 2 "
              "can only make it smaller. Verdict counts only: no model score was read."]
    return "\n".join(lines)


def cmd_mde(args):
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    rows = mde_table(gold, [t for t in args.types.split(",") if t], args.auc, args.r)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mde.json"), "w") as f:
        json.dump(cal.without_nan({"auc_assumed": args.auc, "r": args.r, "rows": rows}), f, indent=2)
        f.write("\n")
    text = mde_markdown(rows, args.auc, args.r)
    print(text)
    return 0


def all_right_n_for_lower(p, z=Z95):
    """Gated items, all of them right, for the Wilson 95% lower bound to reach ``p``: ``n / (n + z^2) >= p``.

    >>> all_right_n_for_lower(0.92)
    45
    """
    return int(math.ceil(z * z * p / (1 - p))) if p < 1 else float("inf")


def needed_table(split_root, shares_path, types=LABEL_TYPES, splits=EVAL_SPLITS, rule=PRIOR_RULE):
    """Per type and evaluation split, from class counts only (no model score): the ``majority2`` prior, the split's
    incorrect share, the Disagree weight ratio, the raw precision a gated Disagree set needs for a prior-weighted
    0.92 (``raw_precision_needed``) and how many gated Disagrees, none wrong, a Wilson 95% lower bound at that raw
    precision needs."""
    rows = []
    for t in types:
        sp = pd.read_csv(os.path.join(split_root, t, "split.csv"), dtype=str, keep_default_na=False)
        pi = load_shares_prior(shares_path, t, rule)
        for s in splits:
            d = sp[sp["split"] == s]
            n_i, n_c = int((d["label_class"] == "incorrect").sum()), int((d["label_class"] == "correct").sum())
            if not (n_i and n_c):
                continue
            ratio = (pi["correct"] / (n_c / (n_c + n_i))) / (pi["incorrect"] / (n_i / (n_c + n_i)))
            need = raw_precision_needed(ratio)
            rows.append({"label_type": t, "split": s, "prior_incorrect": pi["incorrect"],
                         "split_incorrect_share": n_i / (n_c + n_i), "n_incorrect": n_i,
                         "disagree_weight_ratio": ratio, "raw_precision_needed": need,
                         "all_right_n_for_lower_bound": all_right_n_for_lower(need)})
    return rows


def needed_markdown(rows):
    lines = ["| Type | split | majority2 incorrect share | split incorrect share | incorrect rows | raw precision "
             "needed for weighted 0.92 | gated Disagrees, none wrong, for a Wilson 95% lower bound there |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label_type']} | {r['split']} | {r['prior_incorrect']:.3f} | "
                     f"{r['split_incorrect_share']:.3f} | {r['n_incorrect']:,} | {r['raw_precision_needed']:.3f} | "
                     f"{r['all_right_n_for_lower_bound']:,} |")
    lines += ["", "Class counts of the round-3 evaluation splits and the `majority2` prior "
              "(`docs/deployment-prior-2026-11/shares.json`) only; no model score was read."]
    return "\n".join(lines)


def cmd_needed(args):
    rows = needed_table(args.data, args.shares, [t for t in args.types.split(",") if t])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "needed.json"), "w") as f:
        json.dump(cal.without_nan({"rows": rows}), f, indent=2)
        f.write("\n")
    print(needed_markdown(rows))
    return 0


def accounting_markdown(prep_dir, types=LABEL_TYPES):
    """The train-row accounting of the gold-free base and ``cl-drop`` per type, from klone's ``r4_prep.json`` and
    ``variant.json`` (copied to ``prep_dir/<Type>/``). Counts only; no deployment is named."""
    lines = ["| Type | round-3 train rows | gold items dropped | gold scenes dropped (panorama only) | gold-free "
             "train rows (correct / incorrect) | cl-drop train rows (correct / incorrect) | grouping |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for t in types:
        p = os.path.join(prep_dir, t, "r4_prep.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        v = os.path.join(prep_dir, t, "cl", "cl-drop", "variant.json")
        cd = "—"
        base = "—"
        if os.path.exists(v):
            with open(v) as f:
                vv = json.load(f)
            cd = f"{sum(vv['train_after'].values()):,} ({vv['train_after']['correct']:,} / {vv['train_after']['incorrect']:,})"
            base = f"{r['n_train_after']:,} ({vv['train_before']['correct']:,} / {vv['train_before']['incorrect']:,})"
        else:
            base = f"{r['n_train_after']:,}"
        lines.append(f"| {t} | {r['n_train_before']:,} | {r['dropped']['gold_item']} | {r['dropped']['gold_scene']} "
                     f"({r['gold_scene_panorama_only']}) | {base} | {cd} | {r['grouping']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------------------------------------------------

def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _seed_mean(runs, variant, path):
    vals = [_get(r, *path) for r in runs.values() if r.get("variant") == variant and not r.get("missing")]
    vals = [v for v in vals if not _nan(v)]
    return float(np.mean(vals)) if vals else None


def recipe(label_type, mtype, gtype):
    """``(recipe, criteria)``: ``cl-drop`` for the four eligible types unless C1-C4 reject it; else ``baseline``."""
    crit = []
    if label_type not in CL_DROP_TYPES:
        return "baseline", [{"id": "eligible", "result": "fail", "detail": f"{label_type}: cl-drop not eligible"}]
    runs = mtype.get("runs", {})
    # C1: harm on gold (P4 interval wholly below 0).
    p4 = _get(gtype, "p4")
    if p4 is None or _nan(p4.get("hi")):
        crit.append({"id": "C1", "result": "unestimable", "detail": "gold not scored"})
    else:
        crit.append({"id": "C1", "result": "fail" if p4["hi"] < 0 else "pass",
                     "value": p4["mean"], "lo": p4["lo"], "hi": p4["hi"]})
    # C2: in-distribution cost.
    a, b = _seed_mean(runs, "baseline", ("auc", "test")), _seed_mean(runs, "cl-drop", ("auc", "test"))
    if a is None or b is None:
        crit.append({"id": "C2", "result": "unestimable", "detail": "missing test AUC"})
    else:
        crit.append({"id": "C2", "result": "fail" if b < a - C2_MAX_AUC_LOSS else "pass", "baseline": a,
                     "cl-drop": b, "delta": b - a})
    # C3: Agree non-regression (coverage absolute; precision relative to the baseline, A6).
    c3 = {"id": "C3", "result": "pass", "splits": {}}
    for s in EVAL_SPLITS:
        cov = [_seed_mean(runs, v, ("gates", s, "Agree", "coverage_weighted")) for v in ("baseline", "cl-drop")]
        prec = [_seed_mean(runs, v, ("gates", s, "Agree", "precision_weighted")) for v in ("baseline", "cl-drop")]
        row = {"coverage_baseline": cov[0], "coverage_cl-drop": cov[1], "precision_baseline": prec[0],
               "precision_cl-drop": prec[1],
               "baseline_precision_under_gate": None if prec[0] is None else prec[0] < GATE,
               "cl-drop_precision_under_gate": None if prec[1] is None else prec[1] < GATE}
        if None in cov or None in prec:
            row["result"] = "unestimable"
            c3["result"] = "unestimable" if c3["result"] == "pass" else c3["result"]
        else:
            bad = (cov[0] - cov[1] > C3_MAX_COVERAGE_LOSS) or (prec[0] - prec[1] > C3_MAX_PRECISION_LOSS)
            row["result"] = "fail" if bad else "pass"
            if bad:
                c3["result"] = "fail"
        c3["splits"][s] = row
    crit.append(c3)
    # C4: no replication on held-out.
    h = _get(mtype, "contrasts", "cl-drop", "test_heldout")
    if h is None or _nan(h.get("hi")):
        crit.append({"id": "C4", "result": "unestimable", "detail": "no held-out contrast"})
    else:
        crit.append({"id": "C4", "result": "fail" if (h["mean"] < 0 and h["hi"] < 0) else "pass",
                     "value": h["mean"], "lo": h["lo"], "hi": h["hi"]})
    rejected = any(c["result"] == "fail" for c in crit)
    return ("baseline" if rejected else "cl-drop"), crit


def a3_result(block):
    """A3 on one run's gated Disagrees on gold: >= 20 gated: share >= 0.92 and Wilson lower >= 0.80; 1-19: none
    overturned (thin); 0: silent (does not block)."""
    if block is None:
        return "unestimable", "gold not scored"
    n = block["n"]
    if n == 0:
        return "silent", "no gated Disagree on gold"
    if n < A3_MIN_N:
        return ("pass", f"thin: {n} gated, none overturned") if block["overturned"] == 0 else \
            ("fail", f"thin: {block['overturned']} of {n} overturned")
    ok = block["share"] >= A3_SHARE and block["lo"] >= A3_LOWER
    return ("pass" if ok else "fail"), f"{block['confirmed']}/{n} confirmed, lower {block['lo']:.3f}"


def a1_result(run):
    g = _get(run, "gates", "test", "Disagree")
    if g is None:
        return "fail", "no test predictions"
    if g["n_pass"] < A1_MIN_N:
        return "fail", f"{g['n_pass']} gated Disagrees on test (< {A1_MIN_N})"
    pw = g.get("precision_weighted")
    if _nan(pw) or pw < GATE:
        return "fail", f"prior-weighted precision {pw} < {GATE}"
    return "pass", f"{g['n_pass']} gated, weighted precision {pw:.3f}"


def decide_type(label_type, mtype, gtype, ship_seed=SHIP_SEED):
    """One type's record: recipe, criteria, status (never above ``shadow_eligible``) and the reason."""
    rec_variant, rcrit = recipe(label_type, mtype, gtype)
    provisional = any(c["id"] == "C1" and c["result"] == "unestimable" for c in rcrit)
    runs = mtype.get("runs", {})
    ship = runs.get(run_tag(rec_variant, ship_seed), {})
    crit = list(rcrit)
    out = {"label_type": label_type, "recipe": rec_variant,
           "recipe_status": "provisional (gold not scored)" if provisional else "decided"}

    def finish(status, reason):
        assert status in STATUSES
        out.update(status=status, reason=reason, criteria=crit)
        return out

    if not ship or ship.get("missing"):
        return finish("hold_back", "recipe run missing")
    if _nan(_get(ship, "map_max", "Disagree")) or _get(ship, "map_max", "Disagree") < GATE:
        crit.append({"id": "map", "result": "fail", "detail": "the uncapped Disagree map never reaches 0.92"})
        return finish("hold_back", "never clears")
    a1, d1 = a1_result(ship)
    crit.append({"id": "A1", "result": a1, "detail": d1})
    h = _get(ship, "gates", "test_heldout", "Disagree")
    if h is None or h["n_pass"] < A2_MIN_N:
        a2, d2 = "unestimable", "unseen deployments unestimable (Tier B limited to trained deployments)"
    else:
        a2 = "pass" if (not _nan(h["precision_weighted"]) and h["precision_weighted"] >= GATE) else "fail"
        d2 = f"{h['n_pass']} gated, weighted precision {h['precision_weighted']}"
    crit.append({"id": "A2", "result": a2, "detail": d2})
    gship = _get(gtype, "runs", run_tag(rec_variant, ship_seed), "gated_disagree") if gtype else None
    a3, d3 = a3_result(gship)
    crit.append({"id": "A3", "result": a3, "detail": d3})
    if a1 == "fail":
        return finish("hold_back", f"A1: {d1}")
    if a2 == "fail":
        return finish("hold_back", f"A2: {d2}")
    if a3 == "unestimable":
        return finish("hold_back", "gold not scored")
    if a3 == "fail":
        return finish("hold_back", f"A3: {d3}")
    stable = False
    for s in SEEDS:
        if s == ship_seed:
            continue
        r = runs.get(run_tag(rec_variant, s))
        if not r or r.get("missing"):
            continue
        g = _get(gtype, "runs", run_tag(rec_variant, s), "gated_disagree")
        if a1_result(r)[0] == "pass" and a3_result(g)[0] in ("pass", "silent"):
            stable = True
    crit.append({"id": "stability", "result": "pass" if stable else "fail"})
    if not stable:
        return finish("hold_back", "seed-fragile")
    return finish("shadow_eligible", "A1, A2, A3 and stability hold")


def decide(maps, gold=None, ship_seed=SHIP_SEED, types=LABEL_TYPES):
    """Apply the bar to every type in ``maps`` (``maps.json``); ``gold`` is ``gold.json`` or None (pilot)."""
    return [decide_type(t, maps[t], (gold or {}).get(t), ship_seed) for t in types if t in maps]


def cmd_decide(args):
    with open(args.maps) as f:
        maps = json.load(f)
    gold = None
    if args.gold and os.path.exists(args.gold):
        with open(args.gold) as f:
            gold = json.load(f)
        if gold.get("_crops") != "uniform":
            print("gold.json was not built from uniform crops: not used by the bar (A3, C1)", file=sys.stderr)
            gold = None
    res = decide(maps, gold, args.ship_seed)
    with open(args.out, "w") as f:
        json.dump(cal.without_nan(res), f, indent=2, default=float)
        f.write("\n")
    for r in res:
        print(f"{r['label_type']}: recipe {r['recipe']} ({r['recipe_status']}); {r['status']} ({r['reason']})")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def splice(text, blocks):
    """Replace each generated block in ``text``; everything else stays byte-identical. Idempotent.

    >>> t = "a\\n" + BEGIN.format(name="x") + "\\nold\\n" + END.format(name="x") + "\\nb"
    >>> splice(splice(t, {"x": "new"}), {"x": "new"}) == splice(t, {"x": "new"})
    True
    """
    for name, body in blocks.items():
        pat = re.compile(re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)), re.S)
        if not pat.search(text):
            raise ValueError(f"the doc has no generated block {name!r}")
        text = pat.sub(lambda _m: BEGIN.format(name=name) + "\n\n" + body + "\n\n" + END.format(name=name), text)
    return text


def _f(v, fmt="{:.3f}"):
    return "—" if _nan(v) else fmt.format(v)


def maps_markdown(maps):
    lines = ["| Type | run | split | verdict | gated n | weighted coverage | raw precision [95%] | weighted "
             "precision (lower) | raw needed | map max |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in LABEL_TYPES:
        for tag, r in (maps.get(t) or {}).get("runs", {}).items():
            if r.get("missing"):
                lines.append(f"| {t} | {tag} | | | not run | | | | | |")
                continue
            for s in EVAL_SPLITS:
                for v in ("Disagree", "Agree"):
                    g = _get(r, "gates", s, v)
                    if not g:
                        continue
                    lines.append(f"| {t} | {tag} | {s} | {v} | {g['n_pass']} | {_f(g['coverage_weighted'])} | "
                                 f"{_f(g['precision'])} [{_f(g['precision_lo'])}, {_f(g['precision_hi'])}] | "
                                 f"{_f(g['precision_weighted'])} ({_f(g['precision_weighted_lo'])}) | "
                                 f"{_f(g['raw_precision_needed'])} | {_f(_get(r, 'map_max', v))} |")
    return "\n".join(lines)


def bar_markdown(bar):
    lines = ["| Type | recipe | recipe status | status | reason | criteria |", "|---|---|---|---|---|---|"]
    for r in bar:
        crit = "; ".join(f"{c['id']} {c['result']}" for c in r["criteria"])
        lines.append(f"| {r['label_type']} | {r['recipe']} | {r['recipe_status']} | {r['status']} | {r['reason']} | "
                     f"{crit} |")
    return "\n".join(lines)


def gold_markdown(gold):
    lines = []
    if gold.get("_caption"):
        lines += [gold["_caption"], ""]
    lines += ["| Type | run | gated Disagree confirmed | share [95%] | pool-weighted | false alarms (gated / any) | "
              "round-stratified AUC | pooled AUC |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in LABEL_TYPES:
        for tag, r in (gold.get(t) or {}).get("runs", {}).items():
            if r.get("missing"):
                continue
            g = r["gated_disagree"]
            share = "—" if g["n"] < PRIVACY_MIN_CELL else f"{_f(g['share'])} [{_f(g['lo'])}, {_f(g['hi'])}]"
            lines.append(f"| {t} | {tag} | {mask_thin(g['n'], g['confirmed'])} | {share} | "
                         f"{_f(g['share_pool_weighted']) if g['n'] >= PRIVACY_MIN_CELL else '—'} | "
                         f"{_f(r['false_alarm_gated'])} / {_f(r['false_alarm_any'])} | "
                         f"{_f(r['auc_round_stratified'])} | {_f(r['auc_pooled'])} |")
    return "\n".join(lines)


def mde_block(path):
    with open(path) as f:
        d = json.load(f)
    return mde_markdown(d["rows"], d["auc_assumed"], d["r"])


def build_blocks(d):
    blocks = {}
    p = lambda n: os.path.join(d, n)  # noqa: E731
    if os.path.exists(p("mde.json")):
        blocks["mde"] = mde_block(p("mde.json"))
    if os.path.exists(p("needed.json")):
        with open(p("needed.json")) as f:
            blocks["needed"] = needed_markdown(json.load(f)["rows"])
    if os.path.isdir(p("prep")):
        blocks["accounting"] = accounting_markdown(p("prep"))
    if os.path.exists(p("maps.json")):
        with open(p("maps.json")) as f:
            blocks["maps"] = maps_markdown(json.load(f))
    if os.path.exists(p("gold.json")):
        with open(p("gold.json")) as f:
            g = json.load(f)
        blocks["gold_r3crops" if g.get("_crops") == "r3crops" else "gold"] = gold_markdown(g)
    if os.path.exists(p("bar.json")):
        with open(p("bar.json")) as f:
            blocks["bar"] = bar_markdown(json.load(f))
    return blocks


def privacy_gate(text, lists=None):
    """Raise when ``text`` names a non-public deployment (``target_scoreboard.privacy_gate`` over the live lists)."""
    from analysis import target_scoreboard as ts
    lists = lists if lists is not None else ts.city_lists()
    ts.privacy_gate(text, lists, {c: "" for c in lists.get("public", set())})


def cmd_report(args):
    blocks = build_blocks(args.dir)
    if not blocks:
        raise SystemExit(f"nothing to splice from {args.dir}")
    from analysis import target_scoreboard as ts
    lists = ts.city_lists()
    for name, body in blocks.items():
        privacy_gate(body, lists)
    with open(args.doc) as f:
        text = f.read()
    new = splice(text, blocks)
    with open(args.doc, "w") as f:
        f.write(new)
    print(f"spliced {sorted(blocks)} into {args.doc}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    types = ",".join(LABEL_TYPES)
    m = sub.add_parser("maps")
    m.add_argument("--r4", required=True)
    m.add_argument("--shares", default=os.path.join(REPO_DIR, "docs", "deployment-prior-2026-11", "shares.json"))
    m.add_argument("--priors", default=None, help="a deployment_priors.json with a rules block: priors must agree")
    m.add_argument("--types", default=types)
    m.add_argument("--runs", default="baseline,cl-drop,round3")
    m.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    m.add_argument("--prior-rule", default=PRIOR_RULE)
    m.add_argument("--n-boot", type=int, default=N_BOOT)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_maps)
    g = sub.add_parser("gold")
    g.add_argument("--gold", required=True)
    g.add_argument("--pred-dir", required=True, help="<dir>/<Type>/gold_predictions_<crops>_<tag>.csv")
    g.add_argument("--maps", required=True, help="the maps/ directory written by maps")
    g.add_argument("--pool", default=None, help="relabel_pool.csv, for the pool-weighted share")
    g.add_argument("--types", default=types)
    g.add_argument("--runs", default="baseline,cl-drop,round3")
    g.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    g.add_argument("--crops", choices=["uniform", "r3crops"], required=True)
    g.add_argument("--parity", action="store_true")
    g.add_argument("--n-boot", type=int, default=N_BOOT)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gold)
    e = sub.add_parser("mde")
    e.add_argument("--gold", required=True)
    e.add_argument("--types", default=",".join(CL_DROP_TYPES))
    e.add_argument("--auc", type=float, default=0.75)
    e.add_argument("--r", type=float, default=0.8)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_mde)
    n = sub.add_parser("needed")
    n.add_argument("--data", required=True, help="round 3's data dir (<Type>/split.csv)")
    n.add_argument("--shares", default=os.path.join(REPO_DIR, "docs", "deployment-prior-2026-11", "shares.json"))
    n.add_argument("--types", default=types)
    n.add_argument("--out", required=True)
    n.set_defaults(func=cmd_needed)
    d = sub.add_parser("decide")
    d.add_argument("--maps", required=True)
    d.add_argument("--gold", default=None)
    d.add_argument("--ship-seed", type=int, default=SHIP_SEED)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_decide)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--doc", default=os.path.join(REPO_DIR, "docs", "round4-2026-11.md"))
    r.set_defaults(func=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
