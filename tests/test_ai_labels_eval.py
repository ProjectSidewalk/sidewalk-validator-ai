"""The pure parts of analysis/ai_labels_eval.py: vote parsing, the majority reference, campaign tiers, the gate metrics
and the selection on a synthetic rawLabels export. No GPU, no network."""
import doctest
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import ai_labels_eval as ae

AI = ae.dg.AI_USER_ID


def vote(user, v, vt="Human"):
    return {"user_id": user, "validation": v, "validator_type": vt}


def test_doctests():
    failed, _ = doctest.testmod(ae)
    assert failed == 0


def test_vote_counts_ignores_ai_votes_and_counts_experts():
    v = json.dumps([vote("e1", "Agree"), vote("c", "Disagree"), vote("c2", "Unsure"), vote(AI, "Disagree", "AI"),
                    vote("x", "Agree", None)])
    assert ae.vote_counts(v, {"e1"}) == (1, 1, 1, 1, 0)
    assert ae.vote_counts("[]") == (0, 0, 0, 0, 0)
    assert ae.vote_counts(None) == (0, 0, 0, 0, 0)


def test_tier_uses_latest_campaign_on_or_before_the_label_date():
    assert ae.tier_of("laurens-ia", "2026-09-17T01:40:24Z") == "core"
    assert ae.tier_of("laurens-ia", "2026-09-21T23:59:59Z") == "core"
    assert ae.tier_of("laurens-ia", "2026-09-22T01:15:14Z") == "band"
    assert ae.tier_of("vancouver-wa", "2025-09-19") == "core"
    assert ae.tier_of("vancouver-wa", None) == ""


def test_wilson_matches_known_value():
    lo, hi = ae.wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-3) and hi == pytest.approx(0.9784, abs=1e-3)
    assert all(np.isnan(ae.wilson(0, 0)))


def test_gate_metrics_catch_and_precision():
    true = ["incorrect", "incorrect", "incorrect", "correct", "correct"]
    verdict = ["incorrect", "incorrect", "correct", "correct", "incorrect"]
    served = [0.95, 0.90, 0.99, 0.93, 0.95]  # Disagree 1 passes (right), 2 held back, 5 passes (wrong)
    r = ae.gate_metrics(true, verdict, served)
    assert r["disagree"]["n"] == 3 and r["disagree"]["n_pass"] == 2 and r["disagree"]["precision"] == 0.5
    assert r["disagree"]["catch"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["agree"]["n"] == 2 and r["agree"]["n_pass"] == 2 and r["agree"]["precision"] == 0.5
    assert r["all"]["n_pass"] == 4 and r["all"]["coverage"] == 0.8


def test_gate_metrics_no_incorrect_labels():
    r = ae.gate_metrics(["correct"] * 3, ["correct"] * 3, [0.95, 0.5, 0.99])
    assert np.isnan(r["disagree"]["catch"]) and np.isnan(r["disagree"]["precision"])
    assert r["agree"]["n_pass"] == 2


def test_served_accuracy_routes_each_verdict_through_its_class_map():
    api_map = {"correct": {"0.5": 0.0, "0.9": 0.95}, "incorrect": {"0.5": 0.0, "0.9": 0.5}}
    out = ae.served_accuracy(api_map, ["correct", "incorrect", "correct"], [0.95, 0.95, 0.6])
    assert out.tolist() == [0.95, 0.5, 0.0]


def _rawlabels(path, city, rows):
    cols = ["label_id", "user_id", "pano_id", "pano_source", "label_type", "severity", "tags", "description",
            "time_created", "validations", "pano_x", "pano_y", "pano_width", "pano_height", "camera_pitch",
            "camera_roll"]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(path, f"{city}.csv"), index=False)


def test_select_labels_reference_tier_and_round3_join(tmp_path):
    v = lambda *vs: json.dumps([vote(*x) for x in vs])  # noqa: E731
    rows = [
        # label_id, user, pano, source, type, sev, tags, desc, created, validations, x, y, w, h, pitch, roll
        [1, AI, "pA", "mapillary", "CurbRamp", "", "[]", "", "2026-09-05 01:10", v(("c", "Agree")), 10, 20, 100, 50, "", ""],
        [2, AI, "pA", "mapillary", "CurbRamp", "", "[]", "", "2026-09-22 19:03", v(("e1", "Disagree")), 10, 20, 100, 50, "", ""],
        [3, AI, "pB", "mapillary", "CurbRamp", "", "[]", "", "2026-09-22 19:03", v(("c", "Agree"), ("d", "Disagree")), 1, 2, 100, 50, "", ""],
        [4, AI, "pB", "mapillary", "CurbRamp", "", "[]", "", "2026-09-22 19:03", v(("c", "Unsure")), 1, 2, 100, 50, "", ""],
        [5, AI, "pB", "mapillary", "CurbRamp", "", "[]", "", "2026-09-22 19:03", v((AI, "Agree", "AI")), 1, 2, 100, 50, "", ""],
        [6, "h", "pC", "mapillary", "CurbRamp", "2", "[]", "", "2026-09-22 19:03", v(("c", "Agree")), 1, 2, 100, 50, "", ""],
        [7, AI, "pC", "mapillary", "NoCurbRamp", "", "[]", "", "2026-09-22 19:03", v(("c", "Agree")), 1, 2, 100, 50, "", ""],
    ]
    _rawlabels(tmp_path, "richmond-va", rows)
    split = tmp_path / "split.csv"
    pd.DataFrame({"label_uid": ["richmond-va:1"], "split": ["train"]}).to_csv(split, index=False)
    labels, summary = ae.select_labels(str(tmp_path), ["richmond-va"], experts={"e1"}, round3_split=str(split))
    assert labels.label_uid.tolist() == ["richmond-va:1", "richmond-va:2", "richmond-va:6"]
    by = labels.set_index("label_uid")
    assert by.loc["richmond-va:1", ["label_class", "tier", "clean", "round3_split", "author"]].tolist() == \
        ["correct", "core", 0, "train", "ai"]
    assert by.loc["richmond-va:2", ["label_class", "tier", "clean", "expert_class"]].tolist() == \
        ["incorrect", "band", 1, "incorrect"]
    assert by.loc["richmond-va:6", ["author", "tier"]].tolist() == ["human", ""]
    assert summary["ties_dropped"] == 1
    c = summary["per_city"]["richmond-va"]
    assert c["ai_labels_total"] == 5 and c["ai_voted"] == 2 and c["ai_incorrect"] == 1
    assert c["ai_by_tier"] == {"core": 1, "band": 1, "mixed": 0} and c["ai_in_round3"] == {"train": 1}
    assert c["human_voted"] == 1 and c["ai_expert_decided"] == 1
    assert summary["inputs"]["richmond-va"]["sha256"]


def test_auc_block_with_one_class_is_nan():
    df = pd.DataFrame({"true": ["correct"] * 4, "p_incorrect": [0.1, 0.2, 0.3, 0.4], "p_correct": [0.9, 0.8, 0.7, 0.6],
                       "pano_id": ["a", "a", "b", "b"]})
    b = ae.auc_block(df, n_boot=5)
    assert np.isnan(b["auc"]) and b["n"] == 4 and b["acc"] == 1.0


def test_auc_block_perfect_ranking():
    df = pd.DataFrame({"true": ["correct", "correct", "incorrect", "incorrect"], "p_incorrect": [0.1, 0.2, 0.8, 0.9],
                       "p_correct": [0.9, 0.8, 0.2, 0.1], "pano_id": ["a", "b", "c", "d"]})
    b = ae.auc_block(df, n_boot=20)
    assert b["auc"] == 1.0 and b["auc_lo"] == 1.0 and b["recall_incorrect_at_p92"] == 1.0


def test_read_labels_fills_empty_text_cells(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("label_uid,label_id,expert_class,tier,severity,tags\nx:1,1,,core,,[]\nx:2,2,correct,,2,[]\n")
    df = ae.read_labels(str(p))
    assert df.expert_class.tolist() == ["", "correct"] and df.tier.tolist() == ["core", ""]
    assert (df.expert_class != "").sum() == 1  # the NaN would have counted as "decided"
    assert df.label_id.tolist() == ["1", "2"]


def test_gate_catch_is_hits_over_incorrect_not_rounded_precision():
    true = ["incorrect"] * 3 + ["correct"] * 4
    r = ae.gate_metrics(true, ["incorrect"] * 7, [0.95] * 7)
    assert r["disagree"]["catch"] == 1.0  # 3 / 3, not n_pass * round(precision) = 7 * 0.4286 / 3 = 1.0001
    assert r["disagree"]["precision"] == pytest.approx(3 / 7, abs=1e-4)


def test_tier_of_converts_offsets_to_utc():
    assert ae.tier_of("laurens-ia", "2026-09-21T20:00:00-07:00") == "band"   # 03:00 UTC on the 22nd
    assert ae.tier_of("laurens-ia", "2026-09-22T01:00:00+03:00") == "core"   # 22:00 UTC on the 21st
    assert ae.tier_of("laurens-ia", "garbage") == "" and ae.tier_of("laurens-ia", float("nan")) == ""


def test_served_accuracy_exact_key_and_below_first_key():
    api_map = {"correct": {"0.5": 0.1, "0.9": 0.95}, "incorrect": {"0.5": 0.1, "0.9": 0.5}}
    out = ae.served_accuracy(api_map, ["correct", "correct", "incorrect"], [0.9, 0.49, 0.5])
    assert out[0] == 0.95 and out[1] == 0.0 and out[2] == 0.1


def test_rank_score_breaks_saturated_ties():
    s = ae.rank_score([1e-6, 1e-3, 0.5], [1.0, 1.0, 0.5])
    assert s[0] > s[1] > s[2]
    # monotone in p_incorrect where nothing saturates
    s2 = ae.rank_score([0.9, 0.6, 0.2], [0.1, 0.4, 0.8])
    assert s2[0] < s2[1] < s2[2]


def test_auc_block_groups_the_bootstrap_by_pano():
    df = pd.DataFrame({"true": ["correct", "incorrect"] * 20, "p_incorrect": np.r_[np.linspace(0.1, 0.4, 20), np.linspace(0.5, 0.9, 20)][
        np.argsort(np.tile([0, 1], 20), kind="stable")], "pano_id": "p"})
    df["p_correct"] = 1 - df.p_incorrect
    one = ae.auc_block(df, n_boot=50)
    assert one["n_panos"] == 1 and one["auc_lo"] == one["auc_hi"] == one["auc"]  # one pano: every resample is the set
    many = ae.auc_block(df.assign(pano_id=[f"p{i}" for i in range(len(df))]), n_boot=50)
    assert many["auc_lo"] < many["auc_hi"]


def test_recall_at_precision_weighted_matches_unweighted_and_reweights():
    from evaluate import recall_at_precision
    y = [1, 1, 0, 1, 0, 0]
    s = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4]
    assert ae.recall_at_precision_weighted(y, s, [1] * 6, 0.5) == recall_at_precision(y, s, 0.5)
    # weighting negatives 100× makes the top-2 block the only one at precision >= 0.5
    rec, thr = ae.recall_at_precision_weighted(y, s, [1, 1, 100, 1, 100, 100], 0.5)
    assert rec == pytest.approx(2 / 3, abs=1e-4) and thr == 0.8


def test_baseline_at_prevalence_identity_and_reweighting():
    frame = pd.DataFrame({"true": ["incorrect"] * 5 + ["correct"] * 5, "p_incorrect": [0.99, 0.98, 0.97, 0.6, 0.3, 0.96, 0.4, 0.2, 0.1, 0.05]})
    frame["p_correct"] = 1 - frame.p_incorrect
    frame["verdict"] = np.where(frame.p_incorrect > 0.5, "incorrect", "correct")
    r = ae.baseline_at_prevalence(frame, 0.5, 0.75)
    assert r["disagree_precision"] == 0.8 and r["disagree_recall"] == 0.8 and r["recall_incorrect_at_p92"] == 0.8
    r2 = ae.baseline_at_prevalence(frame, 0.05, 0.75)  # one wrong Disagree now weighs 19 right ones
    assert r2["disagree_precision"] == pytest.approx(4 / (4 + 19), abs=1e-4) and r2["disagree_recall"] == 0.8
    assert r2["recall_incorrect_at_p92"] == 0.6  # the top three are clean; the fourth Disagree is the heavy negative
