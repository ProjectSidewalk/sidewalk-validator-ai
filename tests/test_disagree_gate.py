"""analysis/disagree_gate.py: gate statistics for either verdict, the maps, the gold rows, the MDE, the mechanical
decision rules and the report splice (issue #74). Synthetic fixtures only."""
import copy
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from analysis import disagree_gate as dgt
from calibration import calibrate as cal

PI = {"correct": 0.8, "incorrect": 0.2}


# ---------------------------------------------------------------------------------------------------------------------
# gate_stats
# ---------------------------------------------------------------------------------------------------------------------

def frame():
    # 6 correct, 4 incorrect; served >= 0.92 for 3 Disagrees (2 right) and 4 Agrees (3 right)
    return pd.DataFrame({
        "verdict": ["incorrect"] * 4 + ["correct"] * 6,
        "served": [0.95, 0.96, 0.97, 0.5, 0.93, 0.94, 0.95, 0.99, 0.5, 0.6],
        "true": ["incorrect", "incorrect", "correct", "incorrect", "correct", "correct", "correct", "incorrect",
                 "correct", "correct"]})


def test_gate_stats_by_hand_for_both_verdicts():
    d = frame()
    dis = dgt.gate_stats(d, "incorrect", PI)
    # sample mix: 0.6 correct, 0.4 incorrect -> w_correct = 0.8/0.6, w_incorrect = 0.2/0.4
    wc, wi = 0.8 / 0.6, 0.2 / 0.4
    assert (dis["n_pass"], dis["n_right"]) == (3, 2)
    assert dis["precision"] == pytest.approx(2 / 3)
    assert dis["weight_ratio"] == pytest.approx(wc / wi)
    assert dis["precision_weighted"] == pytest.approx(2 * wi / (2 * wi + wc))
    assert dis["coverage_weighted"] == pytest.approx((2 * wi + wc) / (6 * wc + 4 * wi))
    agr = dgt.gate_stats(d, "correct", PI)
    assert (agr["n_pass"], agr["n_right"]) == (4, 3)
    assert agr["weight_ratio"] == pytest.approx(wi / wc)
    assert agr["precision_weighted"] == pytest.approx(3 * wc / (3 * wc + wi))


def test_weighted_bound_is_the_raw_bound_through_the_ratio():
    g = dgt.gate_stats(frame(), "incorrect", PI)
    assert g["precision_weighted_lo"] == pytest.approx(dgt.reweight_precision(g["precision_lo"], g["weight_ratio"]))
    # and raw_precision_needed is the closed form
    r = g["weight_ratio"]
    assert g["raw_precision_needed"] == pytest.approx(0.92 * r / (1 - 0.92 + 0.92 * r))
    assert dgt.reweight_precision(g["raw_precision_needed"], r) == pytest.approx(0.92)


def test_gate_stats_nan_when_a_class_is_absent():
    d = frame()
    d["true"] = "correct"
    g = dgt.gate_stats(d, "correct", PI)
    assert math.isnan(g["precision_weighted"]) and math.isnan(g["weight_ratio"])
    assert g["n_pass"] == 4


# ---------------------------------------------------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------------------------------------------------

def predictions(seed=0, n=600, shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("val", "test", "test_heldout"):
        y = rng.random(n) < 0.35
        p = np.clip(np.where(y, 0.85, 0.15) + rng.normal(0, 0.18, n) + shift, 0.001, 0.999)
        for i in range(n):
            rows.append({"label_uid": f"c:{split}{i}", "split": split, "true": "incorrect" if y[i] else "correct",
                         "p_incorrect": p[i], "p_correct": 1 - p[i]})
    return pd.DataFrame(rows)


def test_margin_reference_row_reproduces_calibrate():
    pred = predictions()
    pi_ref = {"correct": 0.7, "incorrect": 0.3}
    rec, _ = dgt.map_record(pred, PI, pi_ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, rep, _ = cal.calibrate(pred, pi_ref, "val", "test")
    for v, name in (("correct", "Agree"), ("incorrect", "Disagree")):
        mine = rec["reference_margin"]["test"][name]
        assert mine["n_pass"] == rep["eval"][v]["gate_n"]
        if mine["n_pass"]:
            assert mine["precision_weighted"] == pytest.approx(rep["eval"][v]["gate_precision"])


def test_majority2_priors_must_agree(tmp_path):
    p = tmp_path / "deployment_priors.json"
    p.write_text(json.dumps({"Obstacle": {"rule": "margin", "correct": 70, "incorrect": 30,
                                          "rules": {"majority2": {"correct": 80, "incorrect": 20}}}}))
    dgt.check_prior_equal({"correct": 0.8, "incorrect": 0.2}, str(p), "Obstacle")
    with pytest.raises(ValueError, match="differs"):
        dgt.check_prior_equal({"correct": 0.75, "incorrect": 0.25}, str(p), "Obstacle")


def test_paired_seed_delta_is_the_mean_of_per_seed_deltas():
    a = {s: predictions(s) for s in (42, 43)}
    b = {s: predictions(s, shift=0.0) for s in (42, 43)}
    for s in b:
        b[s]["p_incorrect"] = np.where(b[s]["true"] == "incorrect", b[s]["p_incorrect"] + 0.05, b[s]["p_incorrect"])
    groups = {u: u for u in a[42]["label_uid"]}
    out = dgt.paired_seed_delta(a, b, groups, "test_heldout", n_boot=50)
    assert out["mean"] == pytest.approx(np.mean(list(out["per_seed"].values())))
    assert out["mean"] > 0 and out["lo"] <= out["mean"] <= out["hi"]


# ---------------------------------------------------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------------------------------------------------

def gold_table():
    rows = []
    for i in range(12):
        rows.append({"label_uid": f"g:{i}", "label_type": "Obstacle", "round": "1" if i < 6 else "2",
                     "expert": "incorrect" if i % 3 else "correct", "decided": "True", "unmasked": "False",
                     "expert_source": "r1" if i < 6 else "r2_single", "round3_split": "", "ai_vote": "Disagree",
                     "stratum": "lone" if i % 2 else "margin_incorrect", "severity": "2", "tags": "[]"})
    rows[0]["expert_source"] = "r2_consensus"
    rows[1]["round3_split"] = "val"
    rows[2]["decided"] = "False"
    rows[3]["unmasked"] = "True"
    return pd.DataFrame(rows)


def test_gold_population_and_sensitivity_rows():
    g = gold_table()
    decided, head = dgt.gold_population(g)
    assert not head[0] and not head[1] and not decided[2] and not decided[3]
    assert dgt.gold_population(g, consensus_back=True)[1][0]
    assert dgt.gold_population(g, val_back=True)[1][1]
    assert not dgt.gold_population(g, val_back=True, consensus_back=True)[1][2:4].any()   # never counted


def test_gold_join_is_one_to_one():
    g = gold_table()
    pred = pd.DataFrame({"label_uid": ["g:4", "g:4"], "p_correct": [0.1, 0.2], "p_incorrect": [0.9, 0.8]})
    amap = dgt.fit_map(predictions(), PI)
    with pytest.raises(pd.errors.MergeError):
        dgt.gold_rows_for_run(g, pred, amap)


def test_round_stratified_auc_differs_from_pooled_by_hand():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9, 0.85, 0.95])   # each round perfectly ranked, rounds offset
    r = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    assert dgt.stratified_auc(y, s, r) == pytest.approx(1.0)
    assert dgt.auc(y, s) < 1.0


def test_pool_weighted_share_by_hand():
    rows = pd.DataFrame({"round": ["1"] * 4, "label_type": ["T"] * 4, "stratum": ["a", "a", "b", "b"],
                         "verdict": ["incorrect"] * 4, "served": [0.99] * 4,
                         "expert": ["incorrect", "correct", "incorrect", "incorrect"]})
    pool = pd.DataFrame({"label_type": ["T"] * 12, "stratum": ["a"] * 10 + ["b"] * 2})
    w = dgt.pool_weights(rows, {"1": pool})
    assert w.tolist() == [5.0, 5.0, 1.0, 1.0]
    b = dgt.gated_block(rows, "incorrect", w)
    assert b["share"] == pytest.approx(0.75) and b["weighting"] == "pool stratum (per round)"
    assert b["share_pool_weighted"] == pytest.approx((5 + 1 + 1) / 12)
    assert dgt.pool_weights(rows, {"1": pool[pool.stratum == "a"]}) is None   # a stratum without pool counts
    assert dgt.gated_block(rows.assign(served=0.5), "incorrect", w)["weighting"] == "none gated"   # N6
    assert dgt.gated_block(rows, "incorrect", None)["weighting"] == "raw only"


def test_pool_weights_are_per_round():
    # Round 1 and round 2 share stratum names but sampled different frames: a round-2 item takes round 2's counts.
    rows = pd.DataFrame({"round": ["1", "1", "2", "2"], "label_type": ["T"] * 4, "stratum": ["a"] * 4})
    p1 = pd.DataFrame({"label_type": ["T"] * 10, "stratum": ["a"] * 10})
    p2 = pd.DataFrame({"label_type": ["T"] * 3, "stratum": ["a"] * 3})
    assert dgt.pool_weights(rows, {"1": p1, "2": p2}).tolist() == [5.0, 5.0, 1.5, 1.5]
    assert dgt.pool_weights(rows, {"1": p1}) is None                          # round 2 has no pool: raw only
    assert dgt.pool_weights(rows[rows["round"] == "1"], {"1": p1}).tolist() == [5.0, 5.0]


def test_primary_delta_is_the_mean_of_per_seed_deltas():
    g = gold_table().iloc[4:].reset_index(drop=True)
    rng = np.random.default_rng(0)
    rows_a, rows_b = {}, {}
    for s in (42, 43, 44):
        p = rng.random(len(g))
        rows_a[s] = g.assign(p_incorrect=p)
        rows_b[s] = g.assign(p_incorrect=np.where(g["expert"] == "incorrect", p + 0.3, p))
    out = dgt.paired_gold_delta(rows_a, rows_b, n_boot=100)
    assert out["mean"] == pytest.approx(np.mean(list(out["per_seed"].values())))
    for s in (42, 43, 44):
        y = (rows_a[s]["expert"] == "incorrect").astype(int)
        exp = (dgt.stratified_auc(y, rows_b[s]["p_incorrect"], rows_a[s]["round"])
               - dgt.stratified_auc(y, rows_a[s]["p_incorrect"], rows_a[s]["round"]))
        assert out["per_seed"][s] == pytest.approx(exp)


def test_thin_cells_are_masked():
    assert dgt.mask_thin(4, 4) == "n=4" and dgt.mask_thin(5, 4) == "4/5"
    gold = {"_crops": "uniform", "Obstacle": {"runs": {"baseline": {
        "gated_disagree": {"n": 3, "confirmed": 3, "share": 1.0, "lo": 0.4, "hi": 1.0, "share_pool_weighted": 1.0},
        "false_alarm_gated": 0.0, "false_alarm_any": 0.1, "auc_round_stratified": 0.7, "auc_pooled": 0.7}}}}
    text = dgt.gold_markdown(gold)
    assert "n=3" in text and "3/3" not in text and "1.000 [" not in text


def test_r3crops_tables_carry_the_caption():
    text = dgt.gold_markdown({"_crops": "r3crops", "_caption": dgt.R3CROPS_CAPTION})
    assert "not P4, not A3" in text and "confounded" in text


# ---------------------------------------------------------------------------------------------------------------------
# mde
# ---------------------------------------------------------------------------------------------------------------------

def test_mde_matches_the_closed_form():
    a, n1, n0 = 0.75, 60, 40
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    assert dgt.hanley_mcneil_se(a, n1, n0) == pytest.approx(se)
    assert dgt.mde(n1, n0) == pytest.approx((1.959964 + 0.841621) * math.sqrt(2 * se * se * 0.2), rel=1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------------------------------------------------

def gs(n_pass=40, pw=0.95, cov=0.5):
    return {"n_pass": n_pass, "precision_weighted": pw, "coverage_weighted": cov}


def run(variant, seed, dis_test=None, dis_held=None, agree=None, auc_test=0.85, map_max=0.99):
    agree = agree or {}
    return {"variant": variant, "seed": seed, "map_max": {"Disagree": map_max, "Agree": 0.999},
            "auc": {"test": auc_test, "test_heldout": 0.8},
            "gates": {"test": {"Disagree": dis_test or gs(), "Agree": agree.get("test", gs(500, 0.95, 0.6))},
                      "test_heldout": {"Disagree": dis_held or gs(15), "Agree": agree.get("test_heldout",
                                                                                           gs(400, 0.95, 0.6))}}}


def agree_iv(cov=(0.0, -0.02, 0.02), prec=(0.0, -0.005, 0.005), dropped=(0, 0)):
    """One split of maps.json's ``agree_gate`` block (C3's intervals), as ``(mean, lo, hi)`` per clause."""
    out = {"seeds": [42, 43, 44], "n": 1000, "n_boot": 2000, "n_replicates": 2000}
    for k, (m, lo, hi), d in (("coverage", cov, dropped[0]), ("precision", prec, dropped[1])):
        out[k] = {"per_seed": {"42": m, "43": m, "44": m}, "mean": m, "lo": lo, "hi": hi, "n_valid": 2000 - d,
                  "n_dropped": d}
    return out


def maps_ok(t="Obstacle"):
    runs = {}
    for v in ("baseline", "cl-drop"):
        for s in (42, 43, 44):
            runs[dgt.run_tag(v, s)] = run(v, s)
    return {t: {"runs": runs, "contrasts": {"cl-drop": {"test_heldout": {"mean": 0.01, "lo": -0.005, "hi": 0.02}}},
                "agree_gate": {"cl-drop": {s: agree_iv() for s in dgt.EVAL_SPLITS}}}}


def gblock(n=25, overturned=0):
    k = n - overturned
    lo, hi = dgt.wilson(k, n)
    return {"n": n, "confirmed": k, "overturned": overturned, "share": k / n if n else float("nan"), "lo": lo,
            "hi": hi}


def gold_ok(t="Obstacle", block=None):
    runs = {dgt.run_tag(v, s): {"gated_disagree": block or gblock()} for v in ("baseline", "cl-drop")
            for s in (42, 43, 44)}
    return {t: {"runs": runs, "p4": {"mean": 0.01, "lo": -0.05, "hi": 0.06}}}


def one(maps, gold, t="Obstacle"):
    return dgt.decide(maps, gold, types=(t,))[0]


def test_decide_pass():
    r = one(maps_ok(), gold_ok())
    assert r["recipe"] == "cl-drop" and r["status"] == "shadow_eligible" and r["recipe_status"] == "decided"


def test_decide_too_few_gated_on_test():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test"]["Disagree"] = gs(29)
    r = one(m, gold_ok())
    assert r["status"] == "hold_back" and r["reason"].startswith("A1")


def test_decide_weighted_precision_under_gate():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test"]["Disagree"] = gs(40, 0.91)
    assert one(m, gold_ok())["reason"].startswith("A1")


def test_decide_unseen_deployments_contradict_and_unestimable():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test_heldout"]["Disagree"] = gs(12, 0.80)
    assert one(m, gold_ok())["reason"].startswith("A2")
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test_heldout"]["Disagree"] = gs(9, 0.50)
    r = one(m, gold_ok())
    assert r["status"] == "shadow_eligible"                                   # unestimable does not block
    assert any(c["id"] == "A2" and c["result"] == "unestimable" for c in r["criteria"])


def test_decide_gold_contradicts():
    r = one(maps_ok(), gold_ok(block=gblock(25, 4)))
    assert r["status"] == "hold_back" and r["reason"].startswith("A3")


def test_decide_gold_thin_zero_and_one_overturned():
    assert one(maps_ok(), gold_ok(block=gblock(8, 0)))["status"] == "shadow_eligible"
    r = one(maps_ok(), gold_ok(block=gblock(8, 1)))
    assert r["status"] == "hold_back" and "thin" in r["reason"]
    assert one(maps_ok(), gold_ok(block=gblock(0, 0)))["status"] == "shadow_eligible"   # silent does not block


def test_decide_seed_fragile():
    m = maps_ok()
    for s in (43, 44):
        m["Obstacle"]["runs"][f"cl-drop-s{s}"]["gates"]["test"]["Disagree"] = gs(5)
    assert one(m, gold_ok())["reason"] == "seed-fragile"


def test_decide_agree_regression_blocks_the_recipe():
    m = maps_ok()
    for s in (42, 43, 44):
        m["Obstacle"]["runs"][dgt.run_tag("cl-drop", s)]["gates"]["test"]["Agree"] = gs(400, 0.95, 0.55)
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(cov=(-0.05, -0.07, -0.035))   # wholly below -0.03
    r = one(m, gold_ok())
    assert r["recipe"] == "baseline"
    assert any(c["id"] == "C3" and c["result"] == "fail" for c in r["criteria"])


def test_decide_both_arms_under_gate_c3_does_not_block():
    m = maps_ok()
    for v in ("baseline", "cl-drop"):
        for s in (42, 43, 44):
            r_ = m["Obstacle"]["runs"][dgt.run_tag(v, s)]
            r_["gates"]["test"]["Agree"] = gs(400, 0.905 if v == "baseline" else 0.90, 0.6)
    r = one(m, gold_ok())
    c3 = next(c for c in r["criteria"] if c["id"] == "C3")
    assert c3["result"] == "pass" and r["recipe"] == "cl-drop"
    assert c3["splits"]["test"]["baseline_precision_under_gate"] and c3["splits"]["test"]["cl-drop_precision_under_gate"]


def test_decide_c2_and_c4():
    m = maps_ok()
    for s in (42, 43, 44):
        m["Obstacle"]["runs"][dgt.run_tag("cl-drop", s)]["auc"]["test"] = 0.83
    assert one(m, gold_ok())["recipe"] == "baseline"
    m = maps_ok()
    m["Obstacle"]["contrasts"]["cl-drop"]["test_heldout"] = {"mean": -0.02, "lo": -0.03, "hi": -0.01}
    assert one(m, gold_ok())["recipe"] == "baseline"


def test_decide_surfaceproblem_cl_drop_not_eligible():
    m = maps_ok("SurfaceProblem")
    r = one(m, gold_ok("SurfaceProblem"), "SurfaceProblem")
    assert r["recipe"] == "baseline"
    assert r["criteria"][0]["id"] == "eligible"


def test_decide_gold_not_scored():
    r = one(maps_ok(), None)
    assert r["status"] == "hold_back" and r["reason"] == "gold not scored"
    assert r["recipe_status"] == "provisional (gold not scored)"


def test_decide_never_clears_and_never_above_shadow():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["map_max"]["Disagree"] = 0.9199
    assert one(m, gold_ok())["reason"] == "never clears"
    rng = np.random.default_rng(1)
    for _ in range(30):
        m = copy.deepcopy(maps_ok())
        for r_ in m["Obstacle"]["runs"].values():
            r_["gates"]["test"]["Disagree"] = gs(int(rng.integers(0, 200)), float(rng.random()))
            r_["map_max"]["Disagree"] = float(rng.uniform(0.85, 1.0))
        n = int(rng.integers(0, 40))
        g = gold_ok(block=gblock(n, min(n, int(rng.integers(0, 3)))))
        assert one(m, g)["status"] in dgt.STATUSES


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def test_splice_is_idempotent_and_privacy_gate_rejects_private_names():
    t = "intro\n" + dgt.BEGIN.format(name="bar") + "\nold\n" + dgt.END.format(name="bar") + "\nend\n"
    once = dgt.splice(t, {"bar": "| a | b |"})
    assert dgt.splice(once, {"bar": "| a | b |"}) == once and once.startswith("intro\n") and once.endswith("end\n")
    with pytest.raises(ValueError):
        dgt.splice(t, {"missing": "x"})
    lists = {"public": {"seattle-wa"}, "non_public_terms": {"hidden-town"}}
    dgt.privacy_gate("seattle-wa pooled", lists)
    with pytest.raises(ValueError, match="non-public"):
        dgt.privacy_gate("a row for hidden-town", lists)


def test_bar_markdown_lists_every_criterion():
    text = dgt.bar_markdown(dgt.decide(maps_ok(), None, types=("Obstacle",)))
    assert "C3" in text and "A1" in text and "hold_back" in text


# ---------------------------------------------------------------------------------------------------------------------
# gold end to end (B1): the real gold.csv columns, the gold prediction files, per-round pools, parity, P4
# ---------------------------------------------------------------------------------------------------------------------

from analysis import adjudicator as adj  # noqa: E402


def real_gold(n=60, label_type="Obstacle"):
    """A gold table with exactly ``adjudicator.GOLD_COLUMNS``: no ``true`` column, the verdict is ``expert``."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        dis = i % 4 != 3
        rows.append({c: "" for c in adj.GOLD_COLUMNS} | {
            "label_uid": f"c:{i}", "label_type": label_type, "severity": str(1 + i % 3), "tags": "[]",
            "pano_id": f"P{i}", "round": "1" if i < n // 2 else "2",
            "expert": "incorrect" if rng.random() < 0.6 else "correct",
            "expert_source": "r1" if i < n // 2 else "r2_single", "unmasked": "False", "decided": "True",
            "stratum": ("lone" if i % 2 else "margin_incorrect") if dis else "decoy_agree",
            "ai_vote": "Disagree" if dis else "Agree", "round3_split": "test" if i % 5 else "train"})
    return pd.DataFrame(rows, columns=adj.GOLD_COLUMNS)


def write_pool(path, n_lone, n_margin, n_decoy, label_type="Obstacle"):
    rows = ([{"ai_vote": "Disagree", "h_agree": 0, "h_dis": 0}] * n_lone
            + [{"ai_vote": "Disagree", "h_agree": 0, "h_dis": 2}] * n_margin
            + [{"ai_vote": "Agree", "h_agree": 0, "h_dis": 0}] * n_decoy)
    d = pd.DataFrame(rows).assign(schema="x", label_id=range(len(rows)), label_type=label_type, h_unsure=0)
    d.to_csv(path, index=False)
    return str(path)


def gold_layout(tmp_path, gold, runs=(("baseline", 42), ("baseline", 43), ("cl-drop", 42), ("cl-drop", 43),
                                      ("round3", 42))):
    pred_dir, maps_dir = tmp_path / "pred" / "Obstacle", tmp_path / "maps"
    pred_dir.mkdir(parents=True)
    maps_dir.mkdir()
    amap = dgt.fit_map(predictions(), PI)
    y = (gold["expert"] == "incorrect").to_numpy()
    for k, (v, s) in enumerate(runs):
        tag = dgt.run_tag(v, s)
        rng = np.random.default_rng(k)
        p = np.clip(np.where(y, 0.9, 0.2) + rng.normal(0, 0.15, len(y)), 0.001, 0.999)
        for crops, shift in (("uniform", 0.0), ("r3crops", 0.05)):
            pq = np.clip(p + shift, 0.001, 0.999)
            pd.DataFrame({"label_uid": gold["label_uid"], "p_correct": 1 - pq, "p_incorrect": pq}).to_csv(
                pred_dir / f"gold_predictions_{crops}_{tag}.csv", index=False)
        (maps_dir / f"Obstacle_{tag}.json").write_text(json.dumps(amap))
    gold_path = tmp_path / "gold.csv"
    gold.to_csv(gold_path, index=False)
    return gold_path, tmp_path / "pred", maps_dir


def run_gold(gold_path, pred, maps, out, *extra):
    argv = ["gold", "--gold", str(gold_path), "--pred-dir", str(pred), "--maps", str(maps), "--types", "Obstacle",
            "--seeds", "42,43", "--crops", "uniform", "--n-boot", "20", "--out", str(out), *extra]
    assert dgt.main(argv) == 0
    return json.loads((out / "gold.json").read_text())


def test_cmd_gold_end_to_end_with_parity_and_pools(tmp_path):
    gold = real_gold()
    assert "true" not in gold.columns
    gold_path, pred, maps = gold_layout(tmp_path, gold)
    p1 = write_pool(tmp_path / "pool1.csv", 40, 20, 30)
    p2 = write_pool(tmp_path / "pool2.csv", 10, 50, 30)
    res = run_gold(gold_path, pred, maps, tmp_path / "out", "--parity", "--pool", f"1={p1}", "--pool", f"2={p2}")
    ob = res["Obstacle"]
    base = ob["runs"]["baseline"]
    assert base["n_scored"] == ob["populations"]["headline"] == len(gold)
    assert base["gated_disagree"]["weighting"] in ("pool stratum (per round)", "none gated")
    assert set(base["by_round"]) == {"1", "2"} and "1/lone" in base["by_stratum"]
    assert base["parity"]["n"] == len(gold) and base["parity"]["mean_abs_dp"] > 0
    assert ob["p4"]["seeds"] == [42, 43] and "mean" in ob["p4"]
    assert set(ob["ensemble"]) == {"baseline", "cl-drop"} and ob["ensemble"]["baseline"]["seeds"] == [42, 43]
    assert "round3_not_train_rows" in ob["sensitivity"] and "val_back" in ob["sensitivity"]["baseline"]
    prov = res["_provenance"]
    assert str(gold_path) in prov["inputs_sha256"] and p2 in prov["inputs_sha256"]
    assert prov["params"]["n_boot"] == 20 and res["_pool_rounds"] == ["1", "2"]
    # the bar and the report read this file
    maps_json = maps_ok()
    bar = dgt.decide(maps_json, res, types=("Obstacle",))
    assert bar[0]["recipe_status"] == "decided"
    text = dgt.gold_markdown(res)
    assert "**P4**" in text and "Crop parity" in text and "round 1" in text and "Three-seed ensemble" in text


def test_cmd_gold_round_without_pool_is_raw_only(tmp_path):
    gold_path, pred, maps = gold_layout(tmp_path, real_gold())
    p1 = write_pool(tmp_path / "pool1.csv", 40, 20, 30)
    res = run_gold(gold_path, pred, maps, tmp_path / "out", "--pool", f"1={p1}")
    r = res["Obstacle"]["runs"]["baseline"]
    assert r["gated_disagree"]["weighting"] == "raw only"                     # round 2 present, no round-2 pool
    assert r["by_round"]["2"]["gated_disagree"]["weighting"] == "raw only"
    assert r["by_round"]["1"]["gated_disagree"]["weighting"] != "raw only"
    with pytest.raises(SystemExit, match="ROUND=PATH"):
        run_gold(gold_path, pred, maps, tmp_path / "out2", "--pool", p1)
    with pytest.raises(SystemExit, match="missing"):
        run_gold(gold_path, pred, maps, tmp_path / "out3", "--pool", f"1={tmp_path / 'nope.csv'}")


def test_crop_parity_needs_no_truth_column():
    amap = dgt.fit_map(predictions(), PI)
    a = pd.DataFrame({"label_uid": ["x", "y"], "p_correct": [0.9, 0.2], "p_incorrect": [0.1, 0.8]})
    b = a.assign(p_correct=[0.4, 0.2], p_incorrect=[0.6, 0.8])
    out = dgt.crop_parity(a, b, amap)
    assert out["n"] == 2 and out["flip_05"] == pytest.approx(0.5) and out["mean_abs_dp"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------------------------------------------------
# S8: one case per surviving mutant
# ---------------------------------------------------------------------------------------------------------------------

def test_gate_is_inclusive_at_092():
    d = pd.DataFrame({"verdict": ["incorrect", "correct"], "served": [0.92, 0.9199],
                      "true": ["incorrect", "correct"]})
    assert dgt.gate_stats(d, "incorrect", PI)["n_pass"] == 1
    assert dgt.gate_stats(d, "correct", PI)["n_pass"] == 0
    g = d.rename(columns={"true": "expert"})
    assert dgt.gated_block(g, "incorrect")["n"] == 1


def test_a1_a2_a3_boundaries():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test"]["Disagree"] = gs(30)
    assert dgt.a1_result(m["Obstacle"]["runs"]["cl-drop"])[0] == "pass"           # 30 is enough
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test_heldout"]["Disagree"] = gs(10, 0.80)
    assert one(m, gold_ok())["reason"].startswith("A2")                           # 10 is estimable
    assert "thin" not in dgt.a3_result(gblock(20, 0))[1]                         # 20 uses the share rule
    assert "thin" in dgt.a3_result(gblock(19, 0))[1]
    # the lower-bound clause: 23/25 = 0.92 share, Wilson lower ~0.75 < 0.80
    assert dgt.a3_result(gblock(25, 2))[0] == "fail"


def test_a2_one_class_heldout_is_unestimable():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test_heldout"]["Disagree"] = gs(12, float("nan"))
    r = one(m, gold_ok())
    assert any(c["id"] == "A2" and c["result"] == "unestimable" for c in r["criteria"])
    assert r["status"] == "shadow_eligible"


def test_c2_threshold_and_c3_precision_clause_and_c4_upper_bound():
    m = maps_ok()
    for s in (42, 43, 44):
        m["Obstacle"]["runs"][dgt.run_tag("cl-drop", s)]["auc"]["test"] = 0.845      # -0.005: inside 0.010
    r = one(m, gold_ok())
    assert r["recipe"] == "cl-drop" and next(c for c in r["criteria"] if c["id"] == "C2")["result"] == "pass"
    m = maps_ok()
    for s in (42, 43, 44):                                         # same coverage, Agree precision 0.02 lower
        m["Obstacle"]["runs"][dgt.run_tag("cl-drop", s)]["gates"]["test"]["Agree"] = gs(500, 0.93, 0.6)
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(prec=(-0.02, -0.03, -0.011))
    r = one(m, gold_ok())
    assert r["recipe"] == "baseline" and next(c for c in r["criteria"] if c["id"] == "C3")["result"] == "fail"
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(prec=(-0.02, -0.03, -0.01))   # touches -0.01: pass
    assert one(m, gold_ok())["recipe"] == "cl-drop"
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(cov=(-0.05, -0.08, -0.03))    # touches -0.03: pass
    assert one(m, gold_ok())["recipe"] == "cl-drop"
    m = maps_ok()
    m["Obstacle"]["contrasts"]["cl-drop"]["test_heldout"] = {"mean": -0.01, "lo": -0.03, "hi": 0.01}
    assert one(m, gold_ok())["recipe"] == "cl-drop"               # C4 needs the whole interval below 0


def test_c3_coverage_clause_fails_when_cl_drop_gates_no_agree():
    # The final review's case: cl-drop gates no Agree on every seed (coverage 0, precision undefined) while the
    # baseline gates 80%. C3 is an "or", so the coverage loss rejects on its own instead of making C3 unestimable.
    # Since the 2026-11-20 amendment the clauses read the bootstrap: every precision replicate is undefined.
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(0.0, sd) for sd in (42, 43, 44)}
    ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=100, seed=0)
    assert ag["precision"]["n_dropped"] == ag["n_replicates"] == 100 and _isnan(ag["precision"]["hi"])
    assert ag["coverage"]["hi"] < -0.03
    m = maps_ok()
    for v, g in (("baseline", gs(500, 0.95, 0.8)), ("cl-drop", gs(0, float("nan"), 0.0))):
        for s in (42, 43, 44):
            m["Obstacle"]["runs"][dgt.run_tag(v, s)]["gates"]["test"]["Agree"] = dict(g)
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = as_json(ag)
    r = one(m, gold_ok())
    c3 = next(c for c in r["criteria"] if c["id"] == "C3")
    row = c3["splits"]["test"]
    assert row["coverage_result"] == "fail" and row["precision_result"] == "unestimable" and row["result"] == "fail"
    assert c3["result"] == "fail" and r["recipe"] == "baseline"
    assert "test fail (coverage fail, precision unestimable)" in dgt.bar_markdown([r])
    assert "100 replicates undefined" in dgt.bar_markdown([r])
    assert "COLLAPSE FLAG: seed 42: cl-drop gates no Agree" in dgt.bar_markdown([r])
    # recipe() alone, as the reviewer reproduced it
    rec, crit = dgt.recipe("Obstacle", m["Obstacle"], gold_ok()["Obstacle"])
    assert rec == "baseline" and next(c for c in crit if c["id"] == "C3")["result"] == "fail"


def test_c3_undefined_precision_with_kept_coverage_stays_unestimable():
    # Precision interval undefined while coverage holds: no clause fails, one cannot be judged.
    m = maps_ok()
    for v in ("baseline", "cl-drop"):
        for s in (42, 43, 44):
            m["Obstacle"]["runs"][dgt.run_tag(v, s)]["gates"]["test"]["Agree"] = gs(0, float("nan"), 0.0)
    m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(prec=(None, None, None), dropped=(0, 2000))
    r = one(m, gold_ok())
    c3 = next(c for c in r["criteria"] if c["id"] == "C3")
    row = c3["splits"]["test"]
    assert (row["coverage_result"], row["precision_result"], row["result"]) == ("pass", "unestimable", "unestimable")
    assert c3["result"] == "unestimable" and r["recipe"] == "cl-drop"     # unestimable does not reject
    # a precision loss on the other split still fails C3 even though this split is unestimable
    m["Obstacle"]["agree_gate"]["cl-drop"]["test_heldout"] = agree_iv(prec=(-0.05, -0.07, -0.03))
    r = one(m, gold_ok())
    assert next(c for c in r["criteria"] if c["id"] == "C3")["result"] == "fail" and r["recipe"] == "baseline"
    # a maps.json written before the amendment has no agree_gate block: C3 is unestimable, never a silent pass
    m = maps_ok()
    del m["Obstacle"]["agree_gate"]
    c3 = next(c for c in one(m, gold_ok())["criteria"] if c["id"] == "C3")
    assert c3["result"] == "unestimable"


# ---------------------------------------------------------------------------------------------------------------------
# C3's paired hierarchical bootstrap (amended 2026-11-20, J15)
# ---------------------------------------------------------------------------------------------------------------------

N_PANO, PER_PANO = 300, 4
UIDS = [f"c:{i}" for i in range(N_PANO * PER_PANO)]
PANO = {u: f"c|P{i // PER_PANO}" for i, u in enumerate(UIDS)}
TRUE = np.where(np.random.default_rng(0).random(len(UIDS)) < 0.3, "incorrect", "correct")


def agree_scored(cover, seed):
    """One run's served rows on test: every verdict Agree, a row gated (served 0.95) when its run-seeded uniform is
    under ``cover``. Both arms of a training seed share the uniforms, so the paired delta is tight across panoramas
    and all the spread between seeds comes from ``cover``."""
    u = np.random.default_rng(seed).random(len(UIDS))
    return pd.DataFrame({"label_uid": UIDS, "split": "test", "true": TRUE, "verdict": "correct",
                         "served": np.where(u < cover, 0.95, 0.5)})


def as_json(obj):
    return json.loads(json.dumps(cal.without_nan(obj), default=float))


def _isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def with_agree_gate(ag, split="test"):
    m = maps_ok()
    m["Obstacle"]["agree_gate"]["cl-drop"][split] = as_json(ag)
    return m


def test_c3_noisy_seeds_mean_drop_over_003_but_interval_spans_it_passes():
    # mean coverage drop 0.10 (> 0.03, the pre-amendment rule would fail), all of it on one seed of three
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(c, sd) for sd, c in ((42, 0.8), (43, 0.8), (44, 0.5))}
    ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=300, seed=0)
    cov = ag["coverage"]
    assert cov["mean"] < -0.03 and cov["per_seed"][42] == 0.0 and cov["per_seed"][44] < -0.25
    assert cov["lo"] < -0.03 < cov["hi"]
    row = next(c for c in one(with_agree_gate(ag), gold_ok())["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert row["coverage_result"] == "pass" and row["coverage_hi"] == pytest.approx(cov["hi"])


def test_c3_consistent_drop_on_every_seed_fails():
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(0.7, sd) for sd in (42, 43, 44)}
    ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=300, seed=0)
    assert all(v < -0.05 for v in ag["coverage"]["per_seed"].values())
    assert ag["coverage"]["lo"] <= ag["coverage"]["mean"] <= ag["coverage"]["hi"] < -0.03
    r = one(with_agree_gate(ag), gold_ok())
    row = next(c for c in r["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert row["coverage_result"] == "fail" and r["recipe"] == "baseline"


def test_c3_precision_nan_replicates_are_dropped_and_counted():
    # cl-drop gates no Agree on seed 44 only: every replicate that draws seed 44 (1 - 8/27 of them) has an undefined
    # precision and is dropped; under half remain, so the precision clause is unestimable while coverage is judged.
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(c, sd) for sd, c in ((42, 0.8), (43, 0.8), (44, 0.0))}
    ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=300, seed=0)
    prec = ag["precision"]
    assert _isnan(prec["per_seed"][44]) and _isnan(prec["mean"])
    assert prec["n_valid"] + prec["n_dropped"] == ag["n_replicates"] == 300
    assert 0.6 < prec["n_dropped"] / 300 < 0.8 and _isnan(prec["lo"]) and _isnan(prec["hi"])
    assert ag["coverage"]["n_dropped"] == 0 and not _isnan(ag["coverage"]["hi"])
    row = next(c for c in one(with_agree_gate(ag), gold_ok())["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert row["precision_result"] == "unestimable" and row["precision_dropped"] == prec["n_dropped"]
    # no undefined seed: nothing dropped, the interval exists
    ok = dgt.paired_agree_gate_delta(base, base, PANO, "test", PI, n_boot=100, seed=0)
    assert ok["precision"]["n_dropped"] == 0 and ok["precision"]["lo"] == ok["precision"]["hi"] == 0.0


def test_recipe_reads_the_c3_interval_fields():
    ag = {"seeds": [42, 43, 44], "n": 50, "n_boot": 10, "n_replicates": 10,
          "coverage": {"per_seed": {42: -0.2}, "mean": -0.2, "lo": -0.4, "hi": -0.031, "n_valid": 10, "n_dropped": 0},
          "precision": {"per_seed": {42: 0.0}, "mean": 0.0, "lo": -0.012, "hi": 0.01, "n_valid": 7, "n_dropped": 3}}
    rec, crit = dgt.recipe("Obstacle", with_agree_gate(ag)["Obstacle"], gold_ok()["Obstacle"])
    row = next(c for c in crit if c["id"] == "C3")["splits"]["test"]
    assert (row["coverage_delta"], row["coverage_lo"], row["coverage_hi"]) == (-0.2, -0.4, -0.031)
    assert (row["precision_lo"], row["precision_hi"], row["precision_dropped"]) == (-0.012, 0.01, 3)
    assert row["coverage_result"] == "fail" and row["precision_result"] == "pass" and rec == "baseline"
    assert row["coverage_baseline"] == pytest.approx(0.6)          # the seed means stay as descriptive columns
    text = dgt.bar_markdown([one(with_agree_gate(ag), gold_ok())])
    assert "-0.200 [-0.400, -0.031]" in text and "3 replicates undefined" in text


def test_c3_bootstrap_is_deterministic_given_the_seed():
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(c, sd) for sd, c in ((42, 0.75), (43, 0.8), (44, 0.6))}
    a = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=100, seed=7)
    b = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=100, seed=7)
    c = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=100, seed=8)
    assert as_json(a) == as_json(b)
    assert (a["coverage"]["lo"], a["coverage"]["hi"]) != (c["coverage"]["lo"], c["coverage"]["hi"])


def test_c3_point_delta_is_the_difference_of_the_gates_records():
    # The bootstrap's per-seed point value uses the same served verdicts and gate_stats as map_record's gates.
    pi_ref = {"correct": 0.7, "incorrect": 0.3}
    pa = predictions(0)
    pb = pa.assign(p_incorrect=np.clip(pa["p_incorrect"] - 0.04, 0.001, 0.999))
    pb = pb.assign(p_correct=1 - pb["p_incorrect"])
    ra, ma = dgt.map_record(pa, PI, pi_ref)
    rb, mb = dgt.map_record(pb, PI, pi_ref)
    sa, sb = {42: dgt.served_with_map(pa, ma)}, {42: dgt.served_with_map(pb, mb)}
    groups = {u: u for u in pa["label_uid"]}
    for split in dgt.EVAL_SPLITS:
        ag = dgt.paired_agree_gate_delta(sa, sb, groups, split, PI, n_boot=0)
        for k, f in dgt.C3_FIELDS.items():
            want = rb["gates"][split]["Agree"][f] - ra["gates"][split]["Agree"][f]
            assert ag[k]["per_seed"][42] == pytest.approx(want) and ag[k]["mean"] == pytest.approx(want)
        assert _isnan(ag["coverage"]["lo"]) and ag["n_replicates"] == 0          # n_boot 0: no interval


def test_c3_interval_touching_the_margin_passes():
    for kw, limit in (({"cov": (-0.05, -0.08, -0.03)}, "coverage"), ({"prec": (-0.02, -0.03, -0.01)}, "precision")):
        m = maps_ok()
        m["Obstacle"]["agree_gate"]["cl-drop"]["test"] = agree_iv(**kw)
        row = next(c for c in one(m, gold_ok())["criteria"] if c["id"] == "C3")["splits"]["test"]
        assert row[f"{limit}_result"] == "pass" and row[f"{limit}_point_beyond_interval_spans"]


# A panorama-level effect (first review, S2): whole panoramas are ungated together, by both arms (set A) or by cl-drop
# only (set B, shared by every seed), with a little seed-specific row noise in each arm (different rows and rates per
# arm and seed, so pairing cl-drop seed s with baseline seed s+1 changes the interval). Rows within a panorama are
# correlated, the seeds share most of their delta, and the arms share A.
S2_NP, S2_R = 150, 6
S2_UIDS = [f"s:{i:05d}" for i in range(S2_NP * S2_R)]        # zero-padded: sorted order is the fixture's order
S2_GROUPS = {u: f"s|P{i // S2_R}" for i, u in enumerate(S2_UIDS)}
_s2 = np.random.default_rng(5)
S2_TRUE = np.where(_s2.random(len(S2_UIDS)) < 0.3, "incorrect", "correct")
S2_A = _s2.random(S2_NP) < 0.4
S2_B = (_s2.random(S2_NP) < 0.35) & ~S2_A
S2_PANO = np.arange(len(S2_UIDS)) // S2_R
SEEDS3 = (42, 43, 44)


def s2_arm(seed, cl_drop):
    r = np.random.default_rng(seed)
    noise = r.random(len(S2_UIDS)) < 0.02 * (seed - 41)                 # cl-drop only
    base_noise = r.random(len(S2_UIDS)) < 0.01 * (seed - 41)            # baseline only (final review, N-1)
    gated = ~S2_A[S2_PANO] & (~(S2_B[S2_PANO] | noise) if cl_drop else ~base_noise)
    return pd.DataFrame({"label_uid": S2_UIDS, "split": "test", "true": S2_TRUE, "verdict": "correct",
                         "served": np.where(gated, 0.95, 0.5)})


def s2_interval(base, cl, mode, n_boot, seed=0):
    """The coverage interval computed here without the module. ``shared`` is the registered design (draw seeds with
    replacement, one panorama resample per replicate for every drawn seed and both arms, the same rng stream as
    ``paired_agree_gate_delta``); ``per_seed`` and ``per_arm`` are the mistakes the test must catch."""
    from analysis import slices as sl
    y = (S2_TRUE == "incorrect").astype(int)
    g = np.array([S2_GROUPS[u] for u in S2_UIDS])
    rng = np.random.default_rng(seed)
    idxs = sl.boot_indices(y, g, n_boot, rng)
    draws = rng.integers(0, len(SEEDS3), size=(len(idxs), len(SEEDS3)))

    def cov(df, idx):
        return dgt.gate_stats(df.iloc[idx], "correct", PI)["coverage_weighted"]

    vals = []
    for idx, draw in zip(idxs, draws):
        d = []
        for j in draw:
            s = SEEDS3[j]
            if mode == "shared":
                d.append(cov(cl[s], idx) - cov(base[s], idx))
            elif mode == "per_seed":
                own = sl.boot_indices(y, g, 1, rng)[0]
                d.append(cov(cl[s], own) - cov(base[s], own))
            else:                                                                       # per_arm
                d.append(cov(cl[s], sl.boot_indices(y, g, 1, rng)[0]) - cov(base[s], idx))
        vals.append(np.mean(d))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def test_c3_interval_resamples_panoramas_shared_by_every_seed_and_arm():
    base = {s: s2_arm(s, False) for s in SEEDS3}
    cl = {s: s2_arm(s, True) for s in SEEDS3}
    n_boot = 300
    ag = dgt.paired_agree_gate_delta(base, cl, S2_GROUPS, "test", PI, n_boot=n_boot, seed=0)
    lo, hi = ag["coverage"]["lo"], ag["coverage"]["hi"]
    assert (lo, hi) == pytest.approx(s2_interval(base, cl, "shared", n_boot))
    width = hi - lo
    # resampling rows instead of panoramas gives a much narrower interval on this fixture
    rows = dgt.paired_agree_gate_delta(base, cl, {u: u for u in S2_UIDS}, "test", PI, n_boot=n_boot, seed=0)
    assert width > 1.5 * (rows["coverage"]["hi"] - rows["coverage"]["lo"])
    # independent resamples per drawn seed (narrower) or per arm (wider) are each at least 1.4 times off
    for mode in ("per_seed", "per_arm"):
        a, b = s2_interval(base, cl, mode, n_boot)
        assert max(width / (b - a), (b - a) / width) > 1.4, (mode, width, b - a)


def test_c3_row_order_does_not_change_the_result():
    base = {s: s2_arm(s, False) for s in SEEDS3}
    cl = {s: s2_arm(s, True) for s in SEEDS3}
    a = dgt.paired_agree_gate_delta(base, cl, S2_GROUPS, "test", PI, n_boot=60, seed=3)
    shuffled, cl2 = dict(base), dict(cl)
    shuffled[42] = base[42].sample(frac=1.0, random_state=1)        # the first seed's rows, which set the order
    cl2[44] = cl[44].sample(frac=1.0, random_state=2)
    b = dgt.paired_agree_gate_delta(shuffled, cl2, S2_GROUPS, "test", PI, n_boot=60, seed=3)
    assert as_json(a) == as_json(b)


def test_c3_precision_clause_on_a_known_difference():
    # 800 correct, 200 incorrect (the sample mix equals PI, so every prior weight is 1). The baseline gates 400 correct
    # labels; cl-drop gates the same 400 plus 20 incorrect ones: precision 400/420 against 1, delta -0.0476 on every
    # seed; coverage rises by 20/1000.
    n = 1000
    uids = [f"k:{i:05d}" for i in range(n)]
    true = np.array(["correct"] * 800 + ["incorrect"] * 200)
    groups = {u: f"k|P{i // 4}" for i, u in enumerate(uids)}
    base, cl = {}, {}
    for sd in SEEDS3:
        r = np.random.default_rng(sd)
        ok = r.choice(800, 400, replace=False)
        bad = 800 + r.choice(200, 20, replace=False)
        sb, sc = np.full(n, 0.5), np.full(n, 0.5)
        sb[ok] = 0.95
        sc[ok] = 0.95
        sc[bad] = 0.95
        mk = lambda served: pd.DataFrame({"label_uid": uids, "split": "test", "true": true, "verdict": "correct",
                                          "served": served})
        base[sd], cl[sd] = mk(sb), mk(sc)
    ag = dgt.paired_agree_gate_delta(base, cl, groups, "test", PI, n_boot=200, seed=0)
    assert all(v == pytest.approx(400 / 420 - 1) for v in ag["precision"]["per_seed"].values())
    assert all(v == pytest.approx(0.02) for v in ag["coverage"]["per_seed"].values())
    assert ag["precision"]["lo"] <= 400 / 420 - 1 <= ag["precision"]["hi"] < -0.01
    row = next(c for c in one(with_agree_gate(ag), gold_ok())["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert (row["precision_result"], row["coverage_result"], row["result"]) == ("fail", "pass", "fail")


def precision_pair(base_gates):
    """``(baseline, cl-drop)`` scored runs on 1,000 labels (800 correct, 200 incorrect; the mix equals PI, so every
    prior weight is 1). On a seed where ``base_gates[seed]`` is true the baseline gates 400 correct labels (precision
    1.000), otherwise none; cl-drop gates the same 400 plus 60 incorrect ones on every seed (precision 0.870)."""
    n = 1000
    uids = [f"q:{i:05d}" for i in range(n)]
    true = np.array(["correct"] * 800 + ["incorrect"] * 200)
    mk = lambda served: pd.DataFrame({"label_uid": uids, "split": "test", "true": true, "verdict": "correct",
                                      "served": served})
    base, cl = {}, {}
    for sd in SEEDS3:
        r = np.random.default_rng(sd)
        ok, bad = r.choice(800, 400, replace=False), 800 + r.choice(200, 60, replace=False)
        sb, sc = np.full(n, 0.5), np.full(n, 0.5)
        if base_gates[sd]:
            sb[ok] = 0.95
        sc[ok], sc[bad] = 0.95, 0.95
        base[sd], cl[sd] = mk(sb), mk(sc)
    return base, cl, {u: f"q|P{i // 4}" for i, u in enumerate(uids)}


def c3_test_row(ag):
    r = one(with_agree_gate(ag), gold_ok())
    return r, next(c for c in r["criteria"] if c["id"] == "C3")["splits"]["test"]


def test_c3_baseline_gating_no_agree_on_every_seed_reads_as_pass():
    # The baseline gates no Agree on any shared seed; cl-drop gates on every seed: there is no baseline precision to
    # regress from, so the precision clause reads as a pass and only the coverage clause judges.
    base, cl, groups = precision_pair({42: False, 43: False, 44: False})
    ag = dgt.paired_agree_gate_delta(base, cl, groups, "test", PI, n_boot=100, seed=0)
    assert ag["precision"]["undefined"] == {42: "baseline", 43: "baseline", 44: "baseline"}
    assert _isnan(ag["precision"]["hi"]) and dgt.baseline_side_undefined(as_json(ag))
    r, row = c3_test_row(ag)
    assert row["precision_result"] == "pass" and "every shared seed (42, 43, 44" in row["precision_note"]
    assert row["coverage_result"] == "pass" and row["result"] == "pass" and r["recipe"] == "cl-drop"
    assert row["precision_defined_seed_mark"] is None                     # no seed has both arms defined
    assert "precision read as pass: the baseline gates no Agree on every shared seed" in dgt.bar_markdown([r])


def test_c3_baseline_gating_no_agree_on_some_seeds_stays_unestimable_with_the_mark():
    # The final review's case (S-1): the baseline gates no Agree on seed 44 only, while on seeds 42 and 43 it scores
    # 1.000 against cl-drop's 0.870. That loss must not be read as a pass.
    base, cl, groups = precision_pair({42: True, 43: True, 44: False})
    ag = dgt.paired_agree_gate_delta(base, cl, groups, "test", PI, n_boot=100, seed=0)
    prec = ag["precision"]
    assert prec["undefined"] == {44: "baseline"} and _isnan(prec["hi"])
    assert prec["per_seed"][42] == pytest.approx(400 / 460 - 1) and prec["per_seed"][43] == pytest.approx(400 / 460 - 1)
    assert not dgt.baseline_side_undefined(as_json(ag))
    r, row = c3_test_row(ag)
    assert row["precision_result"] == "unestimable" and row["result"] == "unestimable"
    assert row["precision_note"] == ("unestimable: undefined on seed 44 (baseline gates no Agree) of shared seeds "
                                     "42, 43, 44")
    assert row["precision_defined_seed_mark"] == ("precision: defined-seed means differ by more than the margin "
                                                  "(-0.130 over seed 42, 43)")
    text = dgt.bar_markdown([r])
    assert "defined-seed means differ by more than the margin" in text and "does not decide" in text
    assert "read as pass" not in text
    md = dgt.maps_markdown({"Obstacle": {"agree_gate": {"cl-drop": {"test": as_json(ag)}}}})
    assert "defined-seed means differ by more than the margin" in md
    assert "precision undefined on seed 44 (baseline gates no Agree)" in md
    # the mark never decides: the same record with no loss on the defined seeds gives the same recipe and C3
    quiet = as_json(ag)
    quiet["precision"]["per_seed"].update({"42": 0.0, "43": 0.0})
    rec_a, crit_a = dgt.recipe("Obstacle", with_agree_gate(as_json(ag))["Obstacle"], gold_ok()["Obstacle"])
    rec_b, crit_b = dgt.recipe("Obstacle", with_agree_gate(quiet)["Obstacle"], gold_ok()["Obstacle"])
    ra, rb = (next(c for c in crit if c["id"] == "C3")["splits"]["test"] for crit in (crit_a, crit_b))
    assert ra["precision_defined_seed_mark"] and rb["precision_defined_seed_mark"] is None
    assert rec_a == rec_b and [c["result"] for c in crit_a] == [c["result"] for c in crit_b]
    assert (ra["result"], ra["precision_result"]) == (rb["result"], rb["precision_result"])
    # a loss within the margin on the defined seeds prints no mark
    small = as_json(ag)
    small["precision"]["per_seed"].update({"42": -0.01, "43": -0.005})
    assert dgt.defined_seed_mark(small) is None


def test_c3_arm_side_and_mixed_undefined_precision_stay_unestimable_with_the_cause():
    for b44, c44, b43, cause in ((0.8, 0.0, 0.8, "44 (cl-drop gates no Agree)"),
                                 (0.8, 0.0, 0.0, "43 (baseline gates no Agree), 44 (cl-drop gates no Agree)")):
        base = {sd: agree_scored(c, sd) for sd, c in ((42, 0.8), (43, b43), (44, b44))}
        cl = {sd: agree_scored(c, sd) for sd, c in ((42, 0.8), (43, 0.8), (44, c44))}
        ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=50, seed=0)
        _, row = c3_test_row(ag)
        assert row["precision_result"] == "unestimable" and f"undefined on seed {cause}" in row["precision_note"]


def test_c3_prints_the_number_of_shared_seeds_and_warns_under_three():
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    three = dgt.paired_agree_gate_delta(base, base, PANO, "test", PI, n_boot=20, seed=0)
    two = dgt.paired_agree_gate_delta({s: base[s] for s in (42, 43)}, base, PANO, "test", PI, n_boot=20, seed=0)
    one_ = dgt.paired_agree_gate_delta({42: base[42]}, base, PANO, "test", PI, n_boot=20, seed=0)
    r3, row3 = c3_test_row(three)
    r2, row2 = c3_test_row(two)
    r1, row1 = c3_test_row(one_)
    assert (row3["n_seeds"], row2["n_seeds"], row1["n_seeds"]) == (3, 2, 1)
    assert "3 shared seeds" in dgt.bar_markdown([r3]) and "only" not in dgt.bar_markdown([r3])
    assert "only 2 shared seeds: the screen's false-fail rate at the margin is about 25%" in dgt.bar_markdown([r2])
    assert "only 1 shared seed: the interval resamples panoramas only" in dgt.bar_markdown([r1])
    assert r3["recipe"] == r2["recipe"] == r1["recipe"] == "cl-drop"            # a warning, never a floor
    md = dgt.maps_markdown({"Obstacle": {"agree_gate": {"cl-drop": {"test": as_json(two), "test_heldout": three}}}})
    assert "| shared seeds | n |" in md and "only 2 shared seeds" in md
    assert "| none | 3 | 1,200 |" in md and "| 2 | 1,200 |" in md


def test_c3_collapse_flag_and_point_beyond_margin_mark_are_printed_and_do_not_decide():
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(c, sd) for sd, c in ((42, 0.8), (43, 0.8), (44, 0.5))}
    ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=300, seed=0)
    assert ag["collapse"]["flagged"] and ag["collapse"]["seeds"] == [44]            # -0.3 is below -0.09
    r = one(with_agree_gate(ag), gold_ok())
    row = next(c for c in r["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert row["coverage_result"] == "pass" and row["coverage_point_beyond_interval_spans"]
    assert row["collapse"]["seeds"] == [44] and r["recipe"] == "cl-drop"           # descriptive only
    text = dgt.bar_markdown([r])
    assert "COLLAPSE FLAG: seed 44: coverage delta" in text
    assert "coverage: point estimate beyond the margin, interval spans it" in text and "does not decide" in text
    md = dgt.maps_markdown({"Obstacle": {"agree_gate": {"cl-drop": {"test": as_json(ag),
                                                                    "test_heldout": agree_iv()}}}})
    assert "collapse: seed 44: coverage delta" in md and "| none |" in md
    # no flag: a clean pair
    clean = dgt.paired_agree_gate_delta(base, base, PANO, "test", PI, n_boot=20, seed=0)
    assert not clean["collapse"]["flagged"] and "COLLAPSE" not in dgt.bar_markdown([one(with_agree_gate(clean),
                                                                                         gold_ok())])


def test_c3_counts_rows_missing_from_a_seed_and_checks_the_true_class():
    base = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl = {sd: agree_scored(0.8, sd) for sd in (42, 43, 44)}
    cl[43] = cl[43].iloc[5:]                                        # five labels missing from one arm of one seed
    with pytest.warns(UserWarning, match="5 of 1200 labels are missing"):
        ag = dgt.paired_agree_gate_delta(base, cl, PANO, "test", PI, n_boot=20, seed=0)
    assert ag["n_dropped_rows"] == 5 and ag["n"] == len(UIDS) - 5
    row = next(c for c in one(with_agree_gate(ag), gold_ok())["criteria"] if c["id"] == "C3")["splits"]["test"]
    assert row["n_dropped_rows"] == 5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dgt.paired_agree_gate_delta(base, base, PANO, "test", PI, n_boot=5, seed=0)["n_dropped_rows"] == 0
    bad = dict(base)
    bad[43] = agree_scored(0.8, 43)
    bad[43].loc[0, "true"] = "incorrect" if bad[43].loc[0, "true"] == "correct" else "correct"
    with pytest.raises(ValueError, match="true class differs"):
        dgt.paired_agree_gate_delta(base, bad, PANO, "test", PI, n_boot=5, seed=0)


def test_a2_detail_is_rounded():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test_heldout"]["Disagree"] = gs(15, 0.8814216278062682)
    r = one(m, gold_ok())
    a2 = next(c for c in r["criteria"] if c["id"] == "A2")
    assert a2["detail"] == "15 gated, weighted precision 0.881" and "0.8814216" not in dgt.bar_markdown([r])


def test_c2_c3_compare_the_same_seeds():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop-s44"] = {"variant": "cl-drop", "seed": 44, "missing": True}
    m["Obstacle"]["runs"]["baseline-s44"]["auc"]["test"] = 0.50      # would drag the baseline mean down
    r = one(m, gold_ok())
    c2 = next(c for c in r["criteria"] if c["id"] == "C2")
    assert c2["seeds"] == [42, 43] and c2["baseline"] == pytest.approx(0.85)
    c3 = next(c for c in r["criteria"] if c["id"] == "C3")
    assert set(c3["splits"]["test"]["per_seed_coverage"]) == {42, 43}


def test_stability_needs_a3_on_another_seed():
    g = gold_ok()
    for s in (43, 44):
        g["Obstacle"]["runs"][f"cl-drop-s{s}"]["gated_disagree"] = gblock(25, 5)
    assert one(maps_ok(), g)["reason"] == "seed-fragile"


def test_stratified_auc_weights_by_class_pairs():
    # stratum 1: 4 pairs, AUC 1; stratum 2: 1 pair, AUC 0 -> weighted 0.8, the plain mean would be 0.5
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.1])
    r = np.array([1, 1, 1, 1, 2, 2])
    assert dgt.stratified_auc(y, s, r) == pytest.approx(0.8)


def test_gold_auc_reads_only_ai_disagree_items():
    rows = pd.DataFrame({"label_uid": [f"u:{i}" for i in range(6)], "round": ["1"] * 6, "label_type": ["T"] * 6,
                         "stratum": ["a"] * 6,
                         "ai_vote": ["Disagree"] * 4 + ["Agree"] * 2,
                         "expert": ["incorrect", "correct", "incorrect", "correct", "correct", "incorrect"],
                         "p_incorrect": [0.9, 0.1, 0.8, 0.2, 0.99, 0.01], "verdict": "incorrect",
                         "served": 0.5, "severity": "1", "tags": "[]"})
    st = dgt.gold_run_stats(rows)
    assert st["auc_pooled"] == pytest.approx(1.0) and st["n_ai_disagree"] == 4   # the Agree items would break it
    g = rows.assign(round3_split="")
    a = {42: g}
    b = {42: g.assign(p_incorrect=[0.1, 0.9, 0.2, 0.8, 0.0, 1.0])}
    p4 = dgt.paired_gold_delta(a, b, n_boot=0)
    assert p4["n"] == 4 and p4["mean"] == pytest.approx(-1.0)


def test_cmd_decide_ignores_a_non_uniform_gold_file_and_records_provenance(tmp_path):
    mp, gp, out = tmp_path / "maps.json", tmp_path / "gold.json", tmp_path / "bar.json"
    mp.write_text(json.dumps(maps_ok()))
    gp.write_text(json.dumps({"_crops": "r3crops", **gold_ok()}))
    assert dgt.main(["decide", "--maps", str(mp), "--gold", str(gp), "--out", str(out)]) == 0
    bar = json.loads(out.read_text())
    assert bar["types"][0]["reason"] == "gold not scored"
    assert bar["_provenance"]["params"]["gold_used"] is False and str(gp) not in bar["_provenance"]["inputs_sha256"]
    gp.write_text(json.dumps({"_crops": "uniform", **gold_ok()}))
    dgt.main(["decide", "--maps", str(mp), "--gold", str(gp), "--out", str(out)])
    bar = json.loads(out.read_text())
    assert bar["types"][0]["status"] == "shadow_eligible"
    assert bar["_provenance"]["inputs_sha256"][str(gp)] == dgt.sha256_file(str(gp))
    assert "C3" in dgt.bar_markdown(bar) and "per seed" in dgt.bar_markdown(bar)


def test_pilot_bar_without_gold_names_both_reasons():
    m = maps_ok()
    m["Obstacle"]["runs"]["cl-drop"]["gates"]["test"]["Disagree"] = gs(5)
    r = one(m, None)
    assert r["reason"].startswith("A1") and r["reason"].endswith("gold not scored")


def test_cmd_maps_records_provenance_and_contrast(tmp_path):
    r4 = tmp_path / "r4" / "Obstacle"
    r4.mkdir(parents=True)
    for tag, shift in (("baseline", 0.0), ("cl-drop", 0.02), ("round3", -0.02)):   # one test set, one true class
        predictions(0, shift=shift).to_csv(r4 / f"predictions_{tag}.csv", index=False)
    p = predictions(0)
    pd.DataFrame({"label_uid": p["label_uid"], "city": "c", "pano_id": [""] + [f"P{i}" for i in range(len(p) - 1)],
                  "split": p["split"]}).to_csv(r4 / "split.csv", index=False)
    shares = tmp_path / "shares.json"
    shares.write_text(json.dumps({"types": {"Obstacle": {"rules": {"majority2": {"correct": 80, "incorrect": 20},
                                                                   "margin": {"correct": 70, "incorrect": 30}}}}}))
    out = tmp_path / "gate"
    try:
        rc = dgt.main(["maps", "--r4", str(tmp_path / "r4"), "--shares", str(shares), "--types", "Obstacle",
                       "--seeds", "42", "--n-boot", "10", "--out", str(out)])
    except (KeyError, TypeError) as e:          # shares.json layout is prior_rules'; skip if this fixture is off
        pytest.skip(f"shares fixture does not match prior_rules: {e}")
    assert rc == 0
    m = json.loads((out / "maps.json").read_text())
    prov = m["_provenance"]
    assert str(shares) in prov["inputs_sha256"] and str(r4 / "split.csv") in prov["inputs_sha256"]
    assert prov["params"]["n_boot"] == 10 and "git_commit" in prov
    assert "test_heldout" in m["Obstacle"]["contrasts"]["cl-drop"]
    ag = m["Obstacle"]["agree_gate"]["cl-drop"]["test_heldout"]
    assert ag["n_replicates"] == 10 and set(ag) >= {"coverage", "precision", "collapse", "n_dropped_rows"}
    assert ag["coverage"]["lo"] <= ag["coverage"]["hi"]
    text = dgt.maps_markdown(m)
    assert "cl-drop` minus the gold-free baseline" in text and "fit test, eval test_heldout" in text
    assert "C3: Agree at the gate" in text


def test_report_splices_report_files_and_check_mode(tmp_path):
    d = tmp_path / "gate"
    (d / "report").mkdir(parents=True)
    (d / "report" / "slices.md").write_text("# Severity and tag slices: each variant against the round-3 baseline\n\nx")
    (d / "report" / "variants.md").write_text("## Obstacle\n\n| a |")
    doc = tmp_path / "doc.md"
    doc.write_text("".join(f"{dgt.BEGIN.format(name=n)}\nold\n{dgt.END.format(name=n)}\n"
                           for n in ("slices", "variants")))
    lists = {"public": set(), "non_public_terms": set()}
    args = dgt.build_parser().parse_args(["report", "--dir", str(d), "--doc", str(doc), "--check"])
    assert dgt.cmd_report(args, lists) == 1 and "old" in doc.read_text()      # --check writes nothing
    args.check = False
    assert dgt.cmd_report(args, lists) == 0
    text = doc.read_text()
    assert "against the gold-free baseline" in text and "#### Severity" in text and "##### Obstacle" in text
    args.check = True
    assert dgt.cmd_report(args, lists) == 0


# ---------------------------------------------------------------------------------------------------------------------
# P5: symmetric-rb minus matched-rb (amendment of 2026-11-23)
# ---------------------------------------------------------------------------------------------------------------------

def test_p5_agreed_rows_are_passes_rule_symmetric():
    sp = pd.DataFrame({
        "label_uid": [f"a:{i}" for i in range(8)],
        "split": ["test_heldout"] * 7 + ["train"],
        "label_class": ["incorrect", "incorrect", "correct", "incorrect", "incorrect", "incorrect", "correct",
                        "incorrect"],
        "class_source": ["margin", "margin", "margin", "expert", "", "", "margin", "margin"],
        "human_agree": ["0", "0", "3", "0", "0", "1", "4", "0"],
        "human_disagree": ["2", "3", "0", "0", "2", "4", "1", "3"]})
    got = dgt.p5_agreed_rows(sp)
    assert got["label_uid"].tolist() == [f"a:{i}" for i in range(7)]           # test_heldout rows only
    # margin-incorrect at net Disagree 2: not agreed; at 3: agreed; margin-correct at net Agree 3: agreed;
    # expert-incorrect at net Disagree 0: agreed; empty class_source falls back to the votes (net 2 no, net 3 yes)
    assert got["agreed"].tolist() == [False, True, True, True, False, True, True]
    with pytest.raises(ValueError, match="class_source"):
        dgt.p5_agreed_rows(sp.drop(columns="class_source"))


# Panorama fixture for P5 (a): 120 panoramas x 4 rows on test_heldout; 20% expert-sourced; half the crowd-sourced
# incorrect rows at net Disagree 2 (the only rows the two rules disagree on). Both arms of a seed share its noise.
P5_NP, P5_R = 120, 4
P5_UIDS = [f"p:{i:05d}" for i in range(P5_NP * P5_R)]       # zero-padded: sorted order is the fixture's order
P5_PANO = np.arange(len(P5_UIDS)) // P5_R
P5_GROUPS = {u: f"c|P{p}" for u, p in zip(P5_UIDS, P5_PANO)}
_p5 = np.random.default_rng(11)
P5_Y = _p5.random(len(P5_UIDS)) < 0.4
P5_EXPERT = _p5.random(len(P5_UIDS)) < 0.2
P5_DIS2 = P5_Y & ~P5_EXPERT & (_p5.random(len(P5_UIDS)) < 0.5)
P5_BPANO = _p5.random(P5_NP) < 0.3


def p5_split():
    agree = np.where(P5_EXPERT | P5_Y, 0, 4)
    dis = np.where(P5_EXPERT | ~P5_Y, 0, np.where(P5_DIS2, 2, 4))
    return pd.DataFrame({"label_uid": P5_UIDS, "city": "c", "pano_id": [f"P{p}" for p in P5_PANO],
                         "split": "test_heldout", "label_class": np.where(P5_Y, "incorrect", "correct"),
                         "class_source": np.where(P5_EXPERT, "expert", "margin"), "human_agree": agree.astype(str),
                         "human_disagree": dis.astype(str)})


def p5_pred(seed, better=0.0, worse_dis2=0.0, pano_bonus=0.0, noise=0.3):
    """One run's test_heldout predictions: ``better`` raises the incorrect score on the agreed incorrect rows,
    ``worse_dis2`` lowers it on the net-Disagree-2 rows, ``pano_bonus`` raises it on the incorrect rows of a fixed
    set of panoramas (shared by every seed)."""
    e = np.random.default_rng(seed).normal(0, noise, len(P5_UIDS))
    p = (0.5 + np.where(P5_Y, 0.15, -0.15) + e + np.where(P5_Y & ~P5_DIS2, better, 0) - np.where(P5_DIS2, worse_dis2, 0)
         + np.where(P5_Y & P5_BPANO[P5_PANO], pano_bonus, 0))
    return pd.DataFrame({"label_uid": P5_UIDS, "split": "test_heldout", "true": np.where(P5_Y, "incorrect", "correct"),
                         "p_incorrect": p, "p_correct": 1 - p})


def p5_ref_interval(a, b, mode, n_boot, seed=0):
    """P5 (a)'s hierarchical interval computed here without the module (sklearn's AUC, its own alignment).
    ``shared`` is the registered design (C3's draws: panorama resamples first, then seed draws, one panorama
    resample for every drawn seed and both arms); ``per_seed`` and ``per_arm`` are the mistakes the test must
    catch (a fresh panorama resample per drawn seed; the two arms' seeds drawn separately)."""
    from sklearn.metrics import roc_auc_score
    from analysis import slices as sl
    uids = sorted(a[42]["label_uid"])
    y = (a[42].set_index("label_uid").loc[uids, "true"] == "incorrect").astype(int).to_numpy()
    g = np.array([P5_GROUPS[u] for u in uids])
    A = {s: a[s].set_index("label_uid").loc[uids, "p_incorrect"].to_numpy(float) for s in SEEDS3}
    B = {s: b[s].set_index("label_uid").loc[uids, "p_incorrect"].to_numpy(float) for s in SEEDS3}
    rng = np.random.default_rng(seed)
    idxs = sl.boot_indices(y, g, n_boot, rng)
    draws = rng.integers(0, len(SEEDS3), size=(len(idxs), len(SEEDS3)))
    vals = []
    for idx, draw in zip(idxs, draws):
        if mode == "shared":
            vals.append(np.mean([roc_auc_score(y[idx], B[SEEDS3[j]][idx]) - roc_auc_score(y[idx], A[SEEDS3[j]][idx])
                                 for j in draw]))
        elif mode == "per_seed":
            d = []
            for j in draw:
                own = sl.boot_indices(y, g, 1, rng)[0]
                d.append(roc_auc_score(y[own], B[SEEDS3[j]][own]) - roc_auc_score(y[own], A[SEEDS3[j]][own]))
            vals.append(np.mean(d))
        else:                                                                           # per_arm
            draw_b = rng.integers(0, len(SEEDS3), len(SEEDS3))
            vals.append(np.mean([roc_auc_score(y[idx], B[SEEDS3[j]][idx]) for j in draw_b])
                        - np.mean([roc_auc_score(y[idx], A[SEEDS3[j]][idx]) for j in draw]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def test_p5_panorama_hierarchical_interval_matches_an_independent_reference():
    a = {s: p5_pred(s) for s in SEEDS3}
    b = {s: p5_pred(s, pano_bonus=0.3) for s in SEEDS3}
    n_boot = 300
    h = dgt.paired_seed_delta_hierarchical(a, b, P5_GROUPS, "test_heldout", n_boot=n_boot, seed=0)
    assert (h["lo"], h["hi"]) == pytest.approx(p5_ref_interval(a, b, "shared", n_boot), abs=1e-12)
    assert h["n_replicates"] == h["n_valid"] == n_boot
    # the point values are paired_seed_delta's (the registered interval's statistic)
    reg = dgt.paired_seed_delta(a, b, P5_GROUPS, "test_heldout", n_boot=0)
    assert h["mean"] == pytest.approx(reg["mean"]) and h["per_seed"] == pytest.approx(reg["per_seed"])
    width = h["hi"] - h["lo"]
    for mode in ("per_seed", "per_arm"):
        lo, hi = p5_ref_interval(a, b, mode, n_boot)
        assert max(width / (hi - lo), (hi - lo) / width) > 1.3, (mode, width, hi - lo)
    # input order does not change the result
    shuffled = dict(a)
    shuffled[42] = a[42].sample(frac=1.0, random_state=1)
    again = dgt.paired_seed_delta_hierarchical(shuffled, b, P5_GROUPS, "test_heldout", n_boot=n_boot, seed=0)
    assert (again["lo"], again["hi"]) == (h["lo"], h["hi"])


# Gold fixture for P5 (b): 160 items in two rounds, 85% AI-Disagree (P4's rows); b better on a fixed item subset.
G5_N = 160
G5_UIDS = [f"g:{i:04d}" for i in range(G5_N)]
_g5 = np.random.default_rng(21)
G5_Y = _g5.random(G5_N) < 0.55
G5_ROUND = np.where(np.arange(G5_N) < G5_N // 2, "1", "2")
G5_DIS = _g5.random(G5_N) < 0.85
G5_B = _g5.random(G5_N) < 0.3


def g5_rows(seed, bonus=0.0, noise=0.3):
    e = np.random.default_rng(seed + 100).normal(0, noise, G5_N)
    p = 0.5 + np.where(G5_Y, 0.15, -0.15) + e + np.where(G5_Y & G5_B, bonus, 0)
    return pd.DataFrame({"label_uid": G5_UIDS, "ai_vote": np.where(G5_DIS, "Disagree", "Agree"),
                         "expert": np.where(G5_Y, "incorrect", "correct"), "round": G5_ROUND, "p_incorrect": p})


def g5_ref_interval(a, b, mode, n_boot, seed=0):
    """P5 (b)'s hierarchical interval computed here without the module: P4's item resamples first
    (``rng.integers(0, n, n)`` each), then the seed draws; ``per_seed`` / ``per_arm`` are the two mistakes."""
    from sklearn.metrics import roc_auc_score

    def strat(y, s, r):
        num = den = 0.0
        for k in np.unique(r):
            m = r == k
            n1, n0 = y[m].sum(), m.sum() - y[m].sum()
            if n1 and n0:
                num, den = num + roc_auc_score(y[m], s[m]) * n1 * n0, den + n1 * n0
        return num / den if den else float("nan")

    d = a[42][a[42]["ai_vote"] == "Disagree"].sort_values("label_uid")
    uids = list(d["label_uid"])
    y, r = (d["expert"] == "incorrect").astype(int).to_numpy(), d["round"].to_numpy()
    A = {s: a[s].set_index("label_uid").loc[uids, "p_incorrect"].to_numpy(float) for s in SEEDS3}
    B = {s: b[s].set_index("label_uid").loc[uids, "p_incorrect"].to_numpy(float) for s in SEEDS3}
    rng = np.random.default_rng(seed)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    draws = rng.integers(0, len(SEEDS3), size=(n_boot, len(SEEDS3)))
    vals = []
    for idx, draw in zip(idxs, draws):
        if mode == "shared":
            vals.append(np.mean([strat(y[idx], B[SEEDS3[j]][idx], r[idx]) - strat(y[idx], A[SEEDS3[j]][idx], r[idx])
                                 for j in draw]))
        elif mode == "per_seed":
            dd = []
            for j in draw:
                own = rng.integers(0, len(y), len(y))
                dd.append(strat(y[own], B[SEEDS3[j]][own], r[own]) - strat(y[own], A[SEEDS3[j]][own], r[own]))
            vals.append(np.mean(dd))
        else:                                                                           # per_arm
            draw_b = rng.integers(0, len(SEEDS3), len(SEEDS3))
            vals.append(np.mean([strat(y[idx], B[SEEDS3[j]][idx], r[idx]) for j in draw_b])
                        - np.mean([strat(y[idx], A[SEEDS3[j]][idx], r[idx]) for j in draw]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def test_p5_item_hierarchical_interval_matches_an_independent_reference():
    a = {s: g5_rows(s) for s in SEEDS3}
    b = {s: g5_rows(s, bonus=0.3) for s in SEEDS3}
    n_boot = 300
    h = dgt.paired_gold_delta_hierarchical(a, b, n_boot=n_boot, seed=0)
    assert (h["lo"], h["hi"]) == pytest.approx(g5_ref_interval(a, b, "shared", n_boot), abs=1e-12)
    assert h["n"] == int(G5_DIS.sum())                                   # P4's rows: AI-Disagree items only
    reg = dgt.paired_gold_delta(a, b, n_boot=0)                          # the registered interval's statistic
    assert h["mean"] == pytest.approx(reg["mean"]) and h["per_seed"] == pytest.approx(reg["per_seed"])
    width = h["hi"] - h["lo"]
    for mode in ("per_seed", "per_arm"):
        lo, hi = g5_ref_interval(a, b, mode, n_boot)
        assert max(width / (hi - lo), (hi - lo) / width) > 1.3, (mode, width, hi - lo)


def test_p5_sign_is_symmetric_rb_minus_matched_rb():
    matched = {s: p5_pred(s) for s in SEEDS3}
    symmetric = {s: p5_pred(s, better=0.1) for s in SEEDS3}
    leg = dgt.p5_maps_leg(matched, symmetric, p5_split(), n_boot=50)
    assert (leg["a"], leg["b"]) == ("matched-rb", "symmetric-rb")
    for part in ("test_heldout_agreed", "test_heldout"):
        for iv in ("registered", "hierarchical"):
            assert leg[part][iv]["mean"] > 0.03 and all(v > 0 for v in leg[part][iv]["per_seed"].values())
    g = dgt.p5_gold_leg({s: g5_rows(s) for s in SEEDS3}, {s: g5_rows(s, bonus=0.3) for s in SEEDS3}, n_boot=50)
    assert g["registered"]["mean"] > 0 and g["hierarchical"]["mean"] > 0


def test_p5_agreed_rows_mask_the_rows_the_rules_disagree_on():
    # symmetric-rb is better on the agreed incorrect rows and worse on the net-Disagree-2 rows: (a) on the agreed
    # rows reads positive on both intervals, the descriptive every-row contrast spans 0 on both.
    sp = p5_split()
    leg = dgt.p5_maps_leg({s: p5_pred(s) for s in SEEDS3}, {s: p5_pred(s, 0.1, 0.12) for s in SEEDS3}, sp,
                          n_boot=300)
    agreed, every = leg["test_heldout_agreed"], leg["test_heldout"]
    assert agreed["registered"]["lo"] > 0 and agreed["hierarchical"]["lo"] > 0
    for iv in ("registered", "hierarchical"):
        assert every[iv]["lo"] < 0 < every[iv]["hi"]
    n_agreed = int((~P5_DIS2).sum())
    assert (agreed["n"], every["n"]) == (n_agreed, len(P5_UIDS))
    assert agreed["n_incorrect"] == int((P5_Y & ~P5_DIS2).sum()) and agreed["n_correct"] == int((~P5_Y).sum())
    assert agreed["n_panoramas"] == len({P5_PANO[i] for i in np.flatnonzero(~P5_DIS2)})
    assert leg["mde"] == pytest.approx(dgt.mde(agreed["n_incorrect"], agreed["n_correct"]))
    st = dgt.p5_coprimary(leg, "test_heldout_agreed", "a")
    assert st["result"] == "positive"


def test_p5_differing_row_ids_are_an_error_never_intersected():
    sp = p5_split()
    a = {s: p5_pred(s) for s in SEEDS3}
    b = {s: p5_pred(s, better=0.1) for s in SEEDS3}
    b[43] = b[43].iloc[1:]                                               # one row missing from one seed of one arm
    leg = dgt.p5_maps_leg(a, b, sp, n_boot=20)
    assert "row ids differ" in leg["error"] and "test_heldout_agreed" not in leg
    with pytest.raises(ValueError, match="row ids differ"):
        dgt.paired_seed_delta_hierarchical(a, b, P5_GROUPS, "test_heldout", n_boot=10)
    # a scored row whose class differs from split.csv's is an error too
    c = {s: p5_pred(s) for s in SEEDS3}
    c[44] = c[44].copy()
    c[44].loc[0, "true"] = "correct" if P5_Y[0] else "incorrect"
    assert "differs from split.csv" in (dgt.p5_maps_leg(a, c, sp, n_boot=0)["error"] or "")
    # gold: P4 intersects silently (unchanged); P5's gold leg reports an error
    ga = {s: g5_rows(s) for s in SEEDS3}
    gb = {s: g5_rows(s, bonus=0.3) for s in SEEDS3}
    gb[44] = gb[44].iloc[1:]
    assert dgt.paired_gold_delta(ga, gb, n_boot=0)["n"] < int(G5_DIS.sum())
    gleg = dgt.p5_gold_leg(ga, gb, n_boot=0)
    assert "row ids differ" in gleg["error"] and "registered" not in gleg
    with pytest.raises(ValueError, match="row ids differ"):
        dgt.paired_gold_delta_hierarchical(ga, gb, n_boot=0)
    d = dgt.p5_decision({"CurbRamp": {"contrasts": {"symmetric-rb": leg}}}, {"CurbRamp": {"p5": gleg}})
    assert d["answer"] == dgt.P5_UNESTIMABLE
    assert any(c.startswith("CurbRamp (a): error: row ids differ") for c in d["causes"])
    assert any(c.startswith("CurbRamp (b): error: row ids differ") for c in d["causes"])


def p5_iv(lo, hi, mean=None):
    mean = (lo + hi) / 2 if mean is None else mean
    return {"mean": mean, "lo": lo, "hi": hi, "per_seed": {"42": mean, "43": mean, "44": mean}}


def p5_part(reg=(0.01, 0.05), hier=(0.005, 0.06), n=(300, 100)):
    return {"n": sum(n), "n_correct": n[0], "n_incorrect": n[1], "n_panoramas": 200,
            "registered": p5_iv(*reg), "hierarchical": p5_iv(*hier)}


def p5_seeds(a=(42, 43, 44), b=(42, 43, 44)):
    return {"matched-rb": list(a), "symmetric-rb": list(b)}


def p5_files(maps_over=None, gold_over=None):
    """maps.json / gold.json fragments with all four co-primaries positive; ``*_over`` replaces a type's leg."""
    maps = {t: {"contrasts": {"symmetric-rb": {"seeds": p5_seeds(), "error": None, "test_heldout_agreed": p5_part(),
                                               "test_heldout": p5_part((-0.01, 0.02)), "mde": 0.05}}}
            for t in dgt.RB_TYPES}
    gold = {t: {"p5": {"seeds": p5_seeds(), "error": None, **p5_part(n=(50, 60))}} for t in dgt.RB_TYPES}
    for t, leg in (maps_over or {}).items():
        maps[t]["contrasts"]["symmetric-rb"] = leg
    for t, leg in (gold_over or {}).items():
        gold[t]["p5"] = leg
    return as_json(maps), as_json(gold)


def test_p5_decision_needs_all_four_interval_pairs_wholly_above_zero():
    maps, gold = p5_files()
    d = dgt.p5_decision(maps, gold)
    assert d["answer"] == dgt.P5_YES and not d["causes"]
    assert [r["result"] for r in d["rows"] if r["role"] == "co-primary"] == ["positive"] * 4
    assert sum(r["role"] == "descriptive" for r in d["rows"]) == 2           # the every-row contrast never decides
    # registered interval above 0, hierarchical spanning 0: not positive, so the answer is no
    m2, _ = p5_files({"Crosswalk": {**maps["Crosswalk"]["contrasts"]["symmetric-rb"],
                                    "test_heldout_agreed": p5_part(hier=(-0.004, 0.06))}})
    d2 = dgt.p5_decision(m2, gold)
    assert d2["answer"] == dgt.P5_NO and not d2["causes"]
    row = next(r for r in d2["rows"] if r["label_type"] == "Crosswalk" and r["coprimary"] == "a")
    assert row["result"] == "not positive" and row["registered_lo"] > 0
    # three of four positive (CurbRamp's gold leg spans 0 on the registered interval): no
    _, g3 = p5_files(gold_over={"CurbRamp": {"seeds": p5_seeds(), "error": None,
                                             **p5_part(reg=(-0.01, 0.05), n=(50, 60))}})
    d3 = dgt.p5_decision(maps, g3)
    assert d3["answer"] == dgt.P5_NO
    assert [r["result"] for r in d3["rows"] if r["role"] == "co-primary"].count("positive") == 3
    # an interval touching 0 is not wholly above it
    _, g4 = p5_files(gold_over={"Crosswalk": {"seeds": p5_seeds(), "error": None,
                                              **p5_part(hier=(0.0, 0.05), n=(50, 60))}})
    assert dgt.p5_decision(maps, g4)["answer"] == dgt.P5_NO


def test_p5_decision_unestimable_cases_are_never_a_yes():
    maps, gold = p5_files()
    # a missing seed of one arm
    leg = {**maps["CurbRamp"]["contrasts"]["symmetric-rb"], "seeds": p5_seeds(a=(42, 43))}
    d = dgt.p5_decision(p5_files({"CurbRamp": leg})[0], gold)
    assert d["answer"] == dgt.P5_UNESTIMABLE
    assert d["causes"] == ["CurbRamp (a): matched-rb has 2 of the 3 registered seeds"]
    # a thin class on the rows a co-primary is computed on
    _, g = p5_files(gold_over={"Crosswalk": {"seeds": p5_seeds(), "error": None, **p5_part(n=(50, 9))}})
    d = dgt.p5_decision(maps, g)
    assert d["answer"] == dgt.P5_UNESTIMABLE and d["causes"] == ["Crosswalk (b): 9 incorrect rows (fewer than 10)"]
    # no gold at all: both (b) legs unestimable, with the cause named
    d = dgt.p5_decision(maps, None)
    assert d["answer"] == dgt.P5_UNESTIMABLE and sum("(b): gold.json has no P5 entry" in c for c in d["causes"]) == 2
    # neither file has any P5 entry: not run (the pilot)
    assert dgt.p5_decision({"CurbRamp": {"contrasts": {"cl-drop": {}}}}, {"CurbRamp": {}})["answer"] == dgt.P5_NOT_RUN


def test_p5_report_block_not_run_and_with_an_answer():
    assert dgt.p5_markdown({"p5": dgt.p5_decision({}, None)}) == dgt.P5_NOT_RUN_TEXT
    maps, gold = p5_files()
    text = dgt.p5_markdown({"p5": dgt.p5_decision(maps, gold), "_provenance": {"git_commit": "abcdef0123"}})
    assert text.startswith(f"**Answer: {dgt.P5_YES}.**")
    assert "every test_heldout row (descriptive)" in text and "+0.030 [+0.010, +0.050] | [+0.005, +0.060]" in text
    assert "Minimum detectable difference for (a)" in text and "100 incorrect / 300 correct agreed rows" in text
    assert "false yes" in text and "false-fail" not in text and "`abcdef0`" in text
    # the doc carries the block, in its not-run state, so report can splice it once stage 2 has outputs
    with open(os.path.join(dgt.REPO_DIR, "docs", "round4-2026-11.md")) as f:
        doc = f.read()
    blk = re.search(re.escape(dgt.BEGIN.format(name="p5")) + r"\n\n(.*?)\n\n" + re.escape(dgt.END.format(name="p5")),
                    doc, re.S)
    assert blk and blk.group(1) == dgt.P5_NOT_RUN_TEXT


def p5_r4_layout(tmp_path, t="CurbRamp"):
    """``$R4/<t>/`` with split.csv (vote columns) and predictions for every rebalanced run: val / test rows from
    ``predictions(0)``, test_heldout from the P5 fixture; ``symmetric-rb`` better on every seed."""
    d = tmp_path / "r4" / t
    d.mkdir(parents=True)
    vt = predictions(0)
    vt = vt[vt["split"] != "test_heldout"]
    for s in SEEDS3:
        for v, better in (("matched-rb", 0.0), ("symmetric-rb", 0.1)):
            pd.concat([vt, p5_pred(s, better=better)]).to_csv(d / f"predictions_{dgt.run_tag(v, s)}.csv", index=False)
    other = pd.DataFrame({"label_uid": vt["label_uid"], "city": "c", "pano_id": [f"V{i}" for i in range(len(vt))],
                          "split": vt["split"], "label_class": vt["true"], "class_source": "margin",
                          "human_agree": "", "human_disagree": ""})
    pd.concat([other, p5_split()]).to_csv(d / "split.csv", index=False)
    shares = tmp_path / "shares.json"
    shares.write_text(json.dumps({"types": {t: {"rules": {"majority2": {"correct": 80, "incorrect": 20},
                                                          "margin": {"correct": 70, "incorrect": 30}}}}}))
    return tmp_path / "r4", shares


def test_p5_end_to_end_maps_gold_decide_report(tmp_path):
    r4, shares = p5_r4_layout(tmp_path)
    gate = tmp_path / "gate"
    # default --runs: the rebalanced runs are read without being named
    assert dgt.main(["maps", "--r4", str(r4), "--shares", str(shares), "--types", "CurbRamp", "--n-boot", "40",
                     "--out", str(gate)]) == 0
    m = json.loads((gate / "maps.json").read_text())
    leg = m["CurbRamp"]["contrasts"]["symmetric-rb"]
    assert leg["error"] is None and leg["seeds"] == {"matched-rb": [42, 43, 44], "symmetric-rb": [42, 43, 44]}
    assert leg["test_heldout_agreed"]["n"] == int((~P5_DIS2).sum()) and leg["test_heldout"]["n"] == len(P5_UIDS)
    assert leg["test_heldout_agreed"]["registered"]["mean"] > 0                    # symmetric-rb minus matched-rb
    assert leg["test_heldout_agreed"]["hierarchical"]["n_replicates"] == 40
    assert {"symmetric-rb", "matched-rb-s44"} <= set(m["CurbRamp"]["runs"])
    assert str(r4 / "CurbRamp" / "split.csv") in m["_provenance"]["inputs_sha256"]
    # gold: the rebalanced runs have uniform gold predictions but no map; P5 (b) is computed anyway
    gold = real_gold(80, "CurbRamp")
    gold_path, pred = tmp_path / "gold.csv", tmp_path / "pred" / "CurbRamp"
    gold.to_csv(gold_path, index=False)
    pred.mkdir(parents=True)
    y = (gold["expert"] == "incorrect").to_numpy()
    for k, s in enumerate(SEEDS3):
        e = np.random.default_rng(k).normal(0, 0.25, len(y))
        for v, bonus in (("matched-rb", 0.0), ("symmetric-rb", 0.15)):
            p = np.where(y, 0.6 + bonus, 0.4) + e
            pd.DataFrame({"label_uid": gold["label_uid"], "p_correct": 1 - p, "p_incorrect": p}).to_csv(
                pred / f"gold_predictions_uniform_{dgt.run_tag(v, s)}.csv", index=False)
    (tmp_path / "maps").mkdir()
    out = tmp_path / "goldout"
    assert dgt.main(["gold", "--gold", str(gold_path), "--pred-dir", str(tmp_path / "pred"), "--maps",
                     str(tmp_path / "maps"), "--types", "CurbRamp", "--crops", "uniform", "--n-boot", "40",
                     "--out", str(out)]) == 0
    g = json.loads((out / "gold.json").read_text())
    p5 = g["CurbRamp"]["p5"]
    assert p5["error"] is None and p5["registered"]["seeds"] == [42, 43, 44] and p5["registered"]["mean"] > 0
    assert p5["n"] == int(((gold["ai_vote"] == "Disagree")).sum()) and p5["hierarchical"]["n_replicates"] == 40
    assert g["CurbRamp"]["runs"]["symmetric-rb"] == {"missing": True}            # no map: no gate statistics
    # decide and report: Crosswalk has no rebalanced output here, so the answer is unestimable, with the cause
    bar = gate / "bar.json"
    assert dgt.main(["decide", "--maps", str(gate / "maps.json"), "--gold", str(out / "gold.json"),
                     "--out", str(bar)]) == 0
    b = json.loads(bar.read_text())
    assert b["p5"]["answer"] == dgt.P5_UNESTIMABLE
    assert any(c.startswith("Crosswalk (a): maps.json has no P5 entry") for c in b["p5"]["causes"])
    assert not any(c.startswith("CurbRamp") for c in b["p5"]["causes"])
    doc = tmp_path / "doc.md"
    doc.write_text("".join(f"{dgt.BEGIN.format(name=n)}\nold\n{dgt.END.format(name=n)}\n"
                           for n in ("maps", "bar", "p5")))
    args = dgt.build_parser().parse_args(["report", "--dir", str(gate), "--doc", str(doc)])
    assert dgt.cmd_report(args, {"public": set(), "non_public_terms": set()}) == 0
    assert f"**Answer: {dgt.P5_UNESTIMABLE}.**" in doc.read_text()


def test_doctests():
    import doctest
    failed, _ = doctest.testmod(dgt)
    assert failed == 0
