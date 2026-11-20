"""analysis/disagree_gate.py: gate statistics for either verdict, the maps, the gold rows, the MDE, the mechanical
decision rules and the report splice (issue #74). Synthetic fixtures only."""
import copy
import json
import math
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
    rows = pd.DataFrame({"label_type": ["T"] * 4, "stratum": ["a", "a", "b", "b"],
                         "verdict": ["incorrect"] * 4, "served": [0.99] * 4,
                         "expert": ["incorrect", "correct", "incorrect", "incorrect"]})
    pool = pd.DataFrame({"label_type": ["T"] * 12, "stratum": ["a"] * 10 + ["b"] * 2})
    w = dgt.pool_weights(rows, pool)
    assert w.tolist() == [5.0, 5.0, 1.0, 1.0]
    b = dgt.gated_block(rows, "incorrect", w)
    assert b["share"] == pytest.approx(0.75)
    assert b["share_pool_weighted"] == pytest.approx((5 + 1 + 1) / 12)
    assert dgt.pool_weights(rows, pool[pool.stratum == "a"]) is None       # a stratum without pool counts


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


def maps_ok(t="Obstacle"):
    runs = {}
    for v in ("baseline", "cl-drop"):
        for s in (42, 43, 44):
            runs[dgt.run_tag(v, s)] = run(v, s)
    return {t: {"runs": runs, "contrasts": {"cl-drop": {"test_heldout": {"mean": 0.01, "lo": -0.005, "hi": 0.02}}}}}


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
