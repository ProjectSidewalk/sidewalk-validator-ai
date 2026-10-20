"""Tests for analysis/routing.py (human-AI routing simulation) on small synthetic frames.

Schemas are fake (``sidewalk_alpha`` public, ``sidewalk_hidden`` private); nothing is downloaded.
"""
import doctest
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("routing", os.path.join(HERE, "..", "analysis", "routing.py"))
routing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(routing)


def test_doctests():
    assert doctest.testmod(routing).failed == 0


def frame():
    """Ten CurbRamp labels with known scores, verdicts and votes, plus one unlabelled row."""
    rows = [
        # schema, id, verdict, conf, old acc, new acc, oof, h_agree, h_dis
        ("sidewalk_alpha", 1, "Agree", 0.99, 0.95, 0.97, 0.97, 3, 0),     # right
        ("sidewalk_alpha", 2, "Agree", 0.98, 0.94, 0.96, 0.95, 2, 0),     # right
        ("sidewalk_alpha", 3, "Agree", 0.97, 0.93, 0.93, 0.93, 1, 1),     # tie -> 0.5
        ("sidewalk_alpha", 4, "Agree", 0.80, 0.93, 0.85, 0.86, 0, 2),     # wrong
        ("sidewalk_alpha", 5, "Agree", 0.70, 0.90, 0.80, 0.80, 2, 1),     # right
        ("sidewalk_hidden", 1, "Disagree", 0.99, 0.95, 0.60, 0.60, 2, 0),  # wrong (same label_id, other city)
        ("sidewalk_hidden", 2, "Disagree", 0.95, 0.93, 0.55, 0.55, 0, 3),  # right
        ("sidewalk_hidden", 3, "Disagree", 0.90, 0.80, 0.50, 0.50, 2, 2),  # tie
        ("sidewalk_hidden", 4, "Disagree", 0.60, 0.50, 0.40, 0.40, 3, 0),  # wrong
        ("sidewalk_hidden", 5, "Agree", 0.60, 0.50, 0.40, 0.40, 1, 2),     # wrong
        ("sidewalk_alpha", 6, "Agree", 0.99, 0.95, 0.97, np.nan, 1, 0),   # unlabelled (1 vote)
    ]
    df = pd.DataFrame(rows, columns=["schema", "label_id", "ai_result", "ai_confidence", "ai_accuracy", "new_accuracy",
                                     "oof_accuracy", "h_agree", "h_dis"])
    df["label_type"] = "CurbRamp"
    df["h_unsure"] = 0
    df["h_n"] = df.h_agree + df.h_dis
    from calibration.calibrate_from_production import add_outcomes
    return add_outcomes(df, 2, outcome="tie_half")


def test_outcome_matches_calibration_definition():
    df = frame()
    assert df.labelled.sum() == 10
    assert df.loc[2, "correct"] == 0.5 and df.loc[3, "correct"] == 0.0 and df.loc[6, "correct"] == 1.0


def test_sweep_counts_routing_and_both_human_models():
    df = frame()
    lab = df[df.labelled]
    c = routing.sweep(lab.oof_accuracy, lab.correct, [0.92, np.inf], p_h=0.8).set_index("threshold")
    r = c.loc[0.92]
    # AI decides ids 1, 2, 3 (alpha): outcomes 1, 1, 0.5
    assert r.n_ai == 3 and r.routed == pytest.approx(0.7)
    assert r.ai_precision == pytest.approx(2.5 / 3)
    assert r.acc_oracle == pytest.approx((2.5 + 7) / 10)
    assert r.acc_human == pytest.approx((2.5 + 0.8 * 7) / 10)
    none = c.loc[np.inf]
    assert none.n_ai == 0 and none.routed == 1.0 and np.isnan(none.ai_precision) and none.acc_oracle == 1.0


def test_sweep_routed_human_model_uses_loo_on_routed_rows_only():
    s = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    loo = np.array([0.5, np.nan, 0.0, 0.0])                 # only the routed row with a value counts
    r = routing.sweep(s, y, [0.5], p_h=0.9, loo=loo).iloc[0]
    assert r.p_h_routed == pytest.approx(0.5)
    assert r.acc_human_routed == pytest.approx((2 + 0.5 * 2) / 4)


def test_quantile_thresholds_include_gate_and_never():
    th = routing.quantile_thresholds(np.array([0.1, 0.5, 0.9, np.nan]), n_quantiles=4)
    assert 0.92 in th and np.isinf(th[-1]) and 0.1 in th and 0.9 in th


def test_per_verdict_policy_and_pareto():
    df = frame()
    lab = df[df.labelled]
    g = routing.per_verdict_grid(lab, p_h=0.8, n_quantiles=4)
    row = g[(g.t_agree == 0.92) & np.isinf(g.t_disagree)].iloc[0]
    assert row.n_ai == 3 and row.ai_precision == pytest.approx(2.5 / 3)
    td = min(t for t in g.t_disagree.unique() if 0.5 < t <= 0.55)   # a grid value admitting 0.55 and 0.60
    row = g[(g.t_agree == 0.92) & (g.t_disagree == td)].iloc[0]
    # adds Disagree ids 1 (0.60, wrong) and 2 (0.55, right)
    assert row.n_ai == 5 and row.ai_errors == pytest.approx(0.5 + 1)
    assert g.pareto_oracle.any() and g.pareto_human.any()
    # every Pareto point is undominated
    f = g[g.pareto_human]
    for _, p in f.iterrows():
        assert not ((g.routed <= p.routed) & (g.acc_human > p.acc_human + 1e-12)).any()


def test_human_accuracy_from_counts():
    # 3-0: every voter agrees with the other two -> 1; 2-1: (0.5 + 0.5 + 0) / 3
    assert routing.loo_agreement(np.array([3, 2]), np.array([0, 1])).tolist() == pytest.approx([1.0, 1 / 3])
    # AI Disagree on 2-1 Agree: leave out an Agree (2/3) -> tie 0.5; leave out the Disagree (1/3) -> 2-0, AI loses
    assert routing.loo_ai_agreement(np.array([2]), np.array([1]), np.array(["Disagree"]))[0] == pytest.approx(1 / 3)
    t = routing.human_accuracy_table(frame(), 2).set_index("label_type")
    # labels with >= 3 votes: alpha 1 (3-0), alpha 5 (2-1), hidden 2 (0-3), hidden 3 (2-2), hidden 4 (3-0), hidden 5
    assert t.loc["CurbRamp", "n"] == 6


def test_run_marks_the_092_operating_points(tmp_path):
    df = frame()
    res = routing.run(df, str(tmp_path), human_accuracy="0.8", n_boot=5, plots=False)
    ops = res["ops"].set_index("operating_point")
    old = ops.loc["old maps @0.92"]
    # ai_accuracy >= 0.92 on labelled rows: alpha 1-4 and hidden 1-2
    assert old.n_ai == 6
    assert old.ai_precision == pytest.approx((1 + 1 + 0.5 + 0 + 0 + 1) / 6)
    assert old.routed_all_assessed == pytest.approx(4 / 11)
    new = ops.loc["Phase 0 @0.92 (in-sample)"]
    assert new.n_ai == 3 and new.acc_human == pytest.approx((2.5 + 0.8 * 7) / 10)
    assert np.isnan(ops.loc["Phase 0 @0.92 (out-of-fold)", "routed_all_assessed"])
    assert {"curves.csv", "operating_points.csv", "summary.md", "auc.csv"} <= set(os.listdir(tmp_path))


def test_auc_ranks_right_above_wrong():
    assert routing.auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert routing.auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert np.isnan(routing.auc(np.array([0.5]), np.array([1])))


def test_complementarity_removes_the_experts_own_vote():
    df = pd.DataFrame({
        "label_type": ["CurbRamp"] * 3, "ai_result": ["Agree", "Agree", "Disagree"],
        "h_agree": [1, 3, 0], "h_dis": [3, 0, 3], "oof_accuracy": [0.95, 0.95, 0.5],
        # label 0: expert Agree; crowd without the expert is 0-3 -> opposes the AI, expert sides with the AI
        # label 1: the expert did not vote -> not counted
        # label 2: expert Disagree; crowd without the expert is 0-2 -> agrees with the AI
        "expert1": ["Agree", None, "Disagree"]})
    c = routing.complementarity(df, ["expert1"]).set_index(["label_type", "verdict"])
    assert c.loc[("CurbRamp", "Agree"), "n_disagree"] == 1
    assert c.loc[("CurbRamp", "Agree"), "expert_sides_with_ai"] == 1.0
    assert c.loc[("CurbRamp", "Agree"), "n_disagree_above_gate"] == 1
    assert c.loc[("CurbRamp", "Disagree"), "n_agree"] == 1
    assert c.loc[("CurbRamp", "Disagree"), "expert_with_both"] == 1.0


def test_gold_subset_is_three_plus_unanimous():
    g = routing.gold_table(frame(), routing.OPERATING_POINTS)
    g = g[g.label_type == "CurbRamp"].set_index(["verdict", "policy"]).sort_index()
    # Agree gold: alpha 1 (3-0, right). Disagree gold: hidden 2 (0-3, right), hidden 4 (3-0, wrong)
    assert g.loc[("Agree", "all verdicts"), "n_gold"] == 1
    assert g.loc[("Disagree", "all verdicts"), "n_gold"] == 2
    assert g.loc[("Disagree", "all verdicts"), "precision_gold"] == 0.5


def test_public_only_names_and_role_keyed_experts(tmp_path):
    df = frame()
    oof = tmp_path / "labels_oof.csv"
    df.drop(columns=["h_n"]).to_csv(oof, index=False)
    ass = df[["schema", "label_id"]].assign(alice=None, bob=None)
    ass.loc[0, "alice"] = "Agree"
    ass.loc[5, "bob"] = "Disagree"
    ass_path = tmp_path / "ai_assessments.csv"
    ass.to_csv(ass_path, index=False)
    cities = tmp_path / "cities.json"
    cities.write_text(json.dumps([{"city_id": "alpha-wa", "visibility": "public"},
                                  {"city_id": "hidden", "visibility": "private"}]))
    out = tmp_path / "out"
    routing.main(["--labels-oof", str(oof), "--assessments", str(ass_path), "--cities-json", str(cities),
                  "--out-dir", str(out), "--n-boot", "0", "--no-plots"])
    text = "".join(open(out / f).read() for f in os.listdir(out))
    assert "alpha" in text
    assert "hidden" not in text                                 # private deployment never named
    assert "alice" not in text and "bob" not in text            # expert header names never written
    assert "expert1" in (out / "complementarity.csv").read_text()


def test_expert_cols_flag():
    assert routing.parse_expert_cols("b=expert1", ["a", "b"]) == {"b": "expert1"}
    with pytest.raises(ValueError):
        routing.parse_expert_cols("zzz=expert1", ["a", "b"])
