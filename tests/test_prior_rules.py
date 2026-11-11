"""The deployment prior under every vote rule (issue #56): data_generation's rules and entry, calibrate's
--prior-rule, retrain_report's margin keep rate, the dataset card, and analysis/prior_rules.py."""
import json

import numpy as np
import pandas as pd
import pytest

import data_generation as dg
import hf_cards
from analysis import prior_rules as pr
from analysis import retrain_report as rr
from calibration import calibrate as cal
from tests.test_calibrate import _predictions
from tests.test_data_generation import _votes, fixture_labels

AI = dg.AI_USER_ID

# (agree, disagree) -> class under margin, symmetric3, symmetric2, majority2
CASES = {
    (3, 0): ("correct", "correct", "correct", "correct"),
    (2, 0): ("", "", "correct", "correct"),
    (1, 0): ("", "", "", ""),
    (0, 2): ("incorrect", "", "incorrect", "incorrect"),
    (1, 3): ("incorrect", "", "incorrect", "incorrect"),
    (1, 1): ("", "", "", ""),
    (0, 0): ("", "", "", ""),
    (2, 1): ("", "", "", "correct"),
    (0, 3): ("incorrect", "incorrect", "incorrect", "incorrect"),
}


@pytest.mark.parametrize("votes, want", CASES.items())
def test_prior_class_rules(votes, want):
    assert tuple(dg.prior_class(*votes, r) for r in dg.PRIOR_RULES) == want


def test_prior_class_vectorised_and_unknown_rule():
    a, d = zip(*CASES)
    for i, r in enumerate(dg.PRIOR_RULES):
        assert dg.prior_class(pd.Series(a), pd.Series(d), r).tolist() == [w[i] for w in CASES.values()]
    with pytest.raises(ValueError):
        dg.prior_class(1, 0, "symmetric")


def _old_eligible(all_labels, label_type):
    """eligible_labels as it was before the voted_labels refactor (no experts)."""
    df = all_labels[all_labels['label_type'] == label_type]
    keep, dropped = dg.valid_geometry(df)
    df = df[keep].copy()
    df['author'] = df['user_id'].map(dg.label_author) if 'user_id' in df else 'human'
    counts = df['validations'].apply(dg.human_vote_counts)
    df['human_agree'] = [a for a, _ in counts]
    df['human_disagree'] = [d for _, d in counts]
    net = df['human_agree'] - df['human_disagree']
    df['label_class'] = np.where(net > 2, 'correct', np.where(-net > 1, 'incorrect', ''))
    df['class_source'] = np.where(df['label_class'] != '', 'margin', '')
    return df[df['label_class'] != ''].copy(), dropped


def _old_eligible_with_experts(all_labels, label_type, experts):
    """The pre-refactor expert branch, applied to :func:`_old_eligible`'s rows before the class filter."""
    df = all_labels[all_labels['label_type'] == label_type]
    keep, dropped = dg.valid_geometry(df)
    df = df[keep].copy()
    df['author'] = df['user_id'].map(dg.label_author) if 'user_id' in df else 'human'
    counts = df['validations'].apply(dg.human_vote_counts)
    df['human_agree'] = [a for a, _ in counts]
    df['human_disagree'] = [d for _, d in counts]
    net = df['human_agree'] - df['human_disagree']
    df['label_class'] = np.where(net > 2, 'correct', np.where(-net > 1, 'incorrect', ''))
    df['class_source'] = np.where(df['label_class'] != '', 'margin', '')
    verdict = df['validations'].apply(dg.expert_verdict, experts=experts)
    by_expert = (df['label_class'] == '') & (verdict != '')
    df.loc[by_expert, 'label_class'] = verdict[by_expert]
    df.loc[by_expert, 'class_source'] = 'expert'
    return df[df['label_class'] != ''].copy(), dropped


def test_voted_labels_refactor_keeps_eligible_with_experts(tmp_path):
    labels = _prior_fixture(tmp_path)
    experts = dg.parse_experts("someone=ex")
    old, _ = _old_eligible_with_experts(labels, "CurbRamp", experts)
    assert (old.class_source == "expert").any()
    for new, _ in (dg.eligible_labels(labels, "CurbRamp", experts),
                   dg.eligible_labels(labels, "CurbRamp", experts, voted=dg.voted_labels(labels, "CurbRamp"))):
        pd.testing.assert_frame_equal(new, old)
        assert new.to_csv(index=False) == old.to_csv(index=False)


def test_voted_labels_refactor_keeps_eligible(tmp_path):
    labels = fixture_labels(tmp_path)
    old, old_dropped = _old_eligible(labels, "CurbRamp")
    new, dropped = dg.eligible_labels(labels, "CurbRamp")
    voted = dg.voted_labels(labels, "CurbRamp")
    via_voted, _ = dg.eligible_labels(labels, "CurbRamp", voted=voted)
    assert dict(dropped) == dict(old_dropped)
    for frame in (new, via_voted):
        pd.testing.assert_frame_equal(frame, old)
        assert frame.to_csv(index=False) == old.to_csv(index=False)  # byte-identical as written
    assert len(voted[0]) > len(new)  # voted keeps the undecided labels too
    assert "label_class" not in voted[0].columns


def _prior_fixture(tmp_path):
    """fixture_labels plus rows only some rules decide: 2-0 (C, D), 2-1 (D), 0-2 by an AI author (never counted),
    and an expert-only Disagree (never counted)."""
    labels = fixture_labels(tmp_path)
    extra = labels.iloc[:4].copy()
    extra["label_id"] = [201, 202, 203, 204]
    extra["validations"] = [_votes(2, 0), _votes(2, 1), _votes(0, 2), json.dumps(
        [{"user_id": "ex", "validation": "Disagree", "validator_type": "Human"}])]
    extra["user_id"] = ["u", "u", AI, "u"]
    return pd.concat([labels, extra], ignore_index=True)


def test_deployment_prior_entry_legacy_shape(tmp_path):
    eligible, _ = dg.eligible_labels(fixture_labels(tmp_path), "CurbRamp")
    e = dg.deployment_prior_entry(eligible, ["toy-city"], 29, measured_on="2026-10-17")
    assert e == {"correct": 12, "incorrect": 6, "measured_on": "2026-10-17",
                 "source": "rawLabels of toy-city, human votes only", "cities": ["toy-city"], "n_rows": 29}
    with pytest.raises(ValueError, match="needs the voted_labels frame"):
        dg.deployment_prior_entry(eligible, ["toy-city"], 29, rule="majority2")


def test_deployment_prior_entry_rules(tmp_path):
    labels = _prior_fixture(tmp_path)
    experts = dg.parse_experts("someone=ex")
    voted = dg.voted_labels(labels, "CurbRamp")
    eligible, _ = dg.eligible_labels(labels, "CurbRamp", experts, voted=voted)
    assert (eligible.class_source == "expert").sum() == 1  # the expert-only row is in the training pool...
    legacy = dg.deployment_prior_entry(eligible, ["toy-city"], 33, measured_on="x")
    for rule in dg.PRIOR_RULES:
        e = dg.deployment_prior_entry(eligible, ["toy-city"], 33, measured_on="x", voted=voted[0], rule=rule)
        assert e["rule"] == rule and set(e["rules"]) == set(dg.PRIOR_RULES)
        assert e["rules"]["margin"] == {"correct": legacy["correct"], "incorrect": legacy["incorrect"]}
        assert {k: e[k] for k in ("correct", "incorrect")} == e["rules"][rule]
        assert e["cities"] == ["toy-city"] and e["n_rows"] == 33
    rules = e["rules"]
    # ...but not in any prior; nor is the AI-authored 0-2 row. 2-0 counts under symmetric2/majority2, 2-1 under
    # majority2 only.
    assert rules["margin"] == {"correct": 12, "incorrect": 6}
    assert rules["symmetric3"] == {"correct": 12, "incorrect": 6}  # fixture incorrect rows are 0-3
    assert rules["symmetric2"] == {"correct": 13, "incorrect": 6}
    assert rules["majority2"] == {"correct": 14, "incorrect": 6}
    other, _ = dg.eligible_labels(fixture_labels(tmp_path).iloc[:5], "CurbRamp")
    with pytest.raises(ValueError, match="not the same rows"):
        dg.deployment_prior_entry(other, ["toy-city"], 33, voted=voted[0], rule="margin")


def test_main_writes_every_rule_with_majority2_on_top(tmp_path):
    _prior_fixture(tmp_path).to_csv(tmp_path / "toy-city.csv", index=False)
    (tmp_path / dg.CITIES_CACHE_FILE).write_text(json.dumps(
        {"cities": [{"city_id": "toy-city", "visibility": "public", "url": "https://sidewalk-toy.cs.washington.edu"}]}))
    for extra, want in (([], "majority2"), (["--prior-rule", "margin"], "margin")):
        out = tmp_path / f"out-{want}"
        dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path), "--out", str(out),
                 *extra])
        e = json.loads((out / "deployment_priors.json").read_text())["CurbRamp"]
        assert e["rule"] == want and {k: e[k] for k in ("correct", "incorrect")} == e["rules"][want]
        assert e["rules"]["majority2"] == {"correct": 14, "incorrect": 6}


def _priors_file(tmp_path, entry):
    p = tmp_path / "deployment_priors.json"
    p.write_text(json.dumps({"CurbRamp": entry}))
    return p


def test_load_prior_rule_and_fallback(tmp_path):
    new = {"correct": 90, "incorrect": 10, "rule": "majority2",
           "rules": {"margin": {"correct": 60, "incorrect": 40}, "majority2": {"correct": 90, "incorrect": 10}}}
    p = _priors_file(tmp_path, new)
    pi, e = cal.load_prior(p, "CurbRamp")
    assert pi["incorrect"] == pytest.approx(0.1) and e["prior_rule"] == "majority2"
    pi, e = cal.load_prior(p, "CurbRamp", "margin")
    assert pi["incorrect"] == pytest.approx(0.4) and e["prior_rule"] == "margin"
    with pytest.raises(KeyError, match="symmetric3"):
        cal.load_prior(p, "CurbRamp", "symmetric3")
    old = _priors_file(tmp_path, {"correct": 60, "incorrect": 40})  # pre-#56: the top level is the margin rule
    for rule in (None, "margin"):
        pi, e = cal.load_prior(old, "CurbRamp", rule)
        assert pi["incorrect"] == pytest.approx(0.4) and e["prior_rule"] == "margin"
    with pytest.raises(KeyError):
        cal.load_prior(old, "CurbRamp", "majority2")
    face = _priors_file(tmp_path, {"correct": 7, "incorrect": 3, "method": "face_size", "p_correct": 0.7})
    assert cal.load_prior(face, "CurbRamp")[1]["prior_rule"] == "face_size"


def test_report_records_prior_rule(tmp_path):
    preds = pd.concat([_predictions(550, 450, 0.9, 0.8, "val"), _predictions(550, 450, 0.9, 0.8, "test")])
    preds.to_csv(tmp_path / "predictions.csv", index=False)
    p = tmp_path / "priors.json"
    p.write_text(json.dumps({"Obstacle": {"correct": 9, "incorrect": 1, "rule": "majority2", "measured_on": "x",
                                          "source": "y", "rules": {"margin": {"correct": 6, "incorrect": 4},
                                                                   "majority2": {"correct": 9, "incorrect": 1}}}}))
    for extra, rule, inc in (([], "majority2", 0.1), (["--prior-rule", "margin"], "margin", 0.4)):
        out = tmp_path / rule / "obstacle.json"
        cal.main(["--predictions", str(tmp_path / "predictions.csv"), "--priors", str(p), "--label-type",
                  "Obstacle", "--out", str(out), *extra])
        rep = json.loads((out.parent / "calibration_report.json").read_text())
        assert rep["prior_rule"] == rule and rep["prior_source"]["prior_rule"] == rule
        assert rep["prior_deploy"]["incorrect"] == pytest.approx(inc)


def test_correct_weight_uses_margin_counts():
    sp = pd.DataFrame({"label_class": ["correct"] * 2 + ["incorrect"], "author": "human", "class_source": "margin"})
    assert rr.correct_weight(sp, {"correct": 10, "incorrect": 5}) == 5.0  # pre-#56 file
    new = {"correct": 30, "incorrect": 1, "rule": "majority2", "rules": {"margin": {"correct": 10, "incorrect": 5}}}
    assert rr.correct_weight(sp, new) == 5.0
    with pytest.raises(ValueError):
        rr.correct_weight(sp, {"correct": 30, "incorrect": 1, "rule": "majority2"})


def test_card_names_the_face_size_prior():
    text = hf_cards.prior_sentence({"correct": 7, "incorrect": 3, "method": "face_size", "p_correct": 0.7})
    assert "face-size method" in text and "margin" not in text


def test_calibrate_prior_rules_match_data_generation(tmp_path):
    assert cal.PRIOR_RULES == dg.PRIOR_RULES
    with pytest.raises(SystemExit):
        cal.main(["--predictions", "x", "--priors", "y", "--label-type", "CurbRamp", "--out", str(tmp_path / "m.json"),
                  "--prior-rule", "symmetric"])


def test_retrain_report_reweights_to_the_maps_rule(tmp_path):
    p = _priors_file(tmp_path, {"correct": 90, "incorrect": 10, "rule": "majority2",
                                "rules": {"margin": {"correct": 60, "incorrect": 40},
                                          "majority2": {"correct": 90, "incorrect": 10}}})
    for report, inc in (({}, 0.4), ({"prior_rule": "margin"}, 0.4), ({"prior_rule": "majority2"}, 0.1)):
        pi, _ = cal.load_prior(p, "CurbRamp", rr.map_prior_rule(report))
        assert pi["incorrect"] == pytest.approx(inc)


def test_dataset_card_names_the_prior_rule():
    old = hf_cards.prior_sentence({"correct": 60, "incorrect": 40, "source": "s", "measured_on": "d"})
    assert "training margin rule" in old and "0.600 correct" in old
    new = hf_cards.prior_sentence({"correct": 90, "incorrect": 10, "rule": "majority2", "source": "s",
                                   "measured_on": "d", "rules": {"margin": {"correct": 60, "incorrect": 40}}})
    assert "at least two Agree/Disagree votes" in new and "margin rule, under which 60 labels" in new
    assert "expert votes are not counted" in hf_cards.prior_sentence({"correct": 1, "incorrect": 1}, n_experts=2)


# ---------------------------------------------------------------------------------------------------------------------
# analysis/prior_rules.py
# ---------------------------------------------------------------------------------------------------------------------

def test_prior_rules_shares_reproduces_margin(tmp_path):
    labels = _prior_fixture(tmp_path)
    voted = dg.voted_labels(labels, "CurbRamp")
    eligible, _ = dg.eligible_labels(labels, "CurbRamp", voted=voted)
    entry = dg.deployment_prior_entry(eligible, ["toy-city"], 33, voted=voted[0], rule="majority2")
    res = pr.compute_shares(labels, ["CurbRamp"], {"CurbRamp": entry}, min_n=1)
    rec = res["types"]["CurbRamp"]
    assert rec["check"]["ok"] and rec["check"]["n_rows_ok"]
    assert {r: {k: rec["rules"][r][k] for k in ("correct", "incorrect")} for r in dg.PRIOR_RULES} == entry["rules"]
    bad = pr.compute_shares(labels, ["CurbRamp"], {"CurbRamp": {"correct": 1, "incorrect": 1}}, min_n=1)
    assert not bad["types"]["CurbRamp"]["check"]["ok"]
    assert {row["slice"] for row in res["slices"]["CurbRamp"]["tag"]} == {"(none)"}


def test_cmd_shares_hard_check_fails_on_mismatch(tmp_path):
    labels = _prior_fixture(tmp_path)
    labels.to_csv(tmp_path / "toy-city.csv", index=False)
    priors = tmp_path / "priors.json"
    priors.write_text(json.dumps({"CurbRamp": {"correct": 12, "incorrect": 6, "cities": ["toy-city"]}}))
    assert pr.main(["shares", "--rawlabels-cache", str(tmp_path), "--priors", str(priors), "--label-types",
                    "CurbRamp", "--out", str(tmp_path / "ok")]) == 0
    out = json.loads((tmp_path / "ok" / "shares.json").read_text())
    assert "toy-city" not in json.dumps(out) and out["_source"]["n_cities"] == 1  # no city names in the output
    priors.write_text(json.dumps({"CurbRamp": {"correct": 13, "incorrect": 6, "cities": ["toy-city"]}}))
    assert pr.main(["shares", "--rawlabels-cache", str(tmp_path), "--priors", str(priors), "--label-types",
                    "CurbRamp", "--out", str(tmp_path / "bad")]) == 1


def test_gate_join_is_one_to_one():
    pred = pd.DataFrame({"label_uid": ["a:1", "a:2"], "split": "test", "true": "correct", "p_correct": 0.9,
                         "p_incorrect": 0.1})
    split = pd.DataFrame({"label_uid": ["a:1", "a:2"], "severity": ["1", None], "tags": ["[]", '["x"]']})
    assert pr.join_split(pred, split)["severity"].tolist() == ["1", None]
    with pytest.raises(pd.errors.MergeError):
        pr.join_split(pred, pd.concat([split, split.iloc[:1]]))
    with pytest.raises(pd.errors.MergeError):
        pr.join_split(pd.concat([pred, pred.iloc[:1]]), split)
    with pytest.raises(ValueError, match="no split.csv row"):
        pr.join_split(pred, split.iloc[:1])


def test_gate_rows_match_calibrate():
    preds = pd.concat([_predictions(550, 450, 0.9, 0.8, s) for s in ("val", "test", "test_heldout")])
    pi = {"correct": 0.9, "incorrect": 0.1}
    rows = pr.gate_rows(preds, pi)
    _, rep, _ = cal.calibrate(preds, pi, "val", "test_heldout")
    assert rows["test_heldout"]["correct"]["gate_coverage"] == rep["eval"]["correct"]["gate_coverage"]
    report = {"eval_split": "test", "eval": {v: rows["test"][v] for v in cal.CLASSES}, "hold_back": {"verdicts": []}}
    assert pr.check_report(rows["test"], report, "test") == []
    report["eval"] = {v: dict(rows["test"][v], gate_n=rows["test"][v]["gate_n"] + 1) for v in cal.CLASSES}
    assert len(pr.check_report(rows["test"], report, "test")) == 2
    # a missing statistic matches only a missing one: None/NaN on exactly one side is a mismatch
    nan_side = {"gate_coverage": 0.0, "gate_precision": float("nan"), "gate_n": 0}
    none_side = {"gate_coverage": 0.0, "gate_precision": None, "gate_n": 0}
    real_side = {"gate_coverage": 0.0, "gate_precision": 0.95, "gate_n": 0}
    rep = lambda r: {"eval_split": "test", "eval": {"correct": r, "incorrect": r}, "hold_back": {"verdicts": []}}
    mine = {"correct": nan_side, "incorrect": nan_side}
    assert pr.check_report(mine, rep(none_side), "test") == []
    assert len(pr.check_report(mine, rep(real_side), "test")) == 2
    assert len(pr.check_report({"correct": real_side, "incorrect": real_side}, rep(none_side), "test")) == 2


def test_observed_keys_by_schema_and_label_and_splits_periods(tmp_path):
    rows = [  # schema, label_id, type, deleted, result, accuracy, date, agree, disagree
        ("sidewalk_a", 1, "CurbRamp", "f", "Agree", 0.95, "2026-10-30", 2, 0),
        ("sidewalk_b", 1, "CurbRamp", "f", "Agree", 0.95, "2026-10-30", 0, 2),  # same label_id, other city
        ("sidewalk_a", 2, "CurbRamp", "f", "Agree", 0.90, "2026-10-30", 3, 0),  # below the gate
        ("sidewalk_a", 3, "CurbRamp", "f", "Agree", 0.95, "2026-10-27", 3, 0),  # changeover day
        ("sidewalk_a", 4, "CurbRamp", "f", "Agree", 0.95, "2026-10-01", 2, 1),  # older model
        ("sidewalk_a", 5, "CurbRamp", "t", "Agree", 0.95, "2026-10-30", 3, 0),  # deleted
        ("sidewalk_a_bak_2026_11", 1, "CurbRamp", "f", "Agree", 0.95, "2026-10-30", 3, 0),  # backup schema
        ("sidewalk_a", 6, "Obstacle", "f", "Agree", 0.95, "2026-10-30", 3, 0),  # retyped: CurbRamp model's vote
    ]
    d = pd.DataFrame(rows, columns=["schema", "label_id", "label_type", "deleted", "ai_result", "ai_accuracy",
                                    "ai_date", "h_agree", "h_dis"])
    d["model_id"] = "sidewalk-validator-ai-curbramp"
    path = tmp_path / "a.csv"
    d.to_csv(path, index=False)
    df, counts = pr.load_assessments(path)
    assert counts["backup_schema_rows"] == 1 and counts["deleted_rows"] == 1 and counts["model_type_mismatch"] == 1
    assert not ((df.label_type == "Obstacle")).any()
    assert counts["periods"] == {"round3": 3, "changeover": 1, "older": 1}
    res = pr.observed_tables(df)
    r3 = res["gated_agree"]["round3"]["CurbRamp"]
    assert r3["n"] == 2 and {k: r3["majority2"][k] for k in ("decided", "confirmed", "share_confirmed")} == \
        {"decided": 2, "confirmed": 1, "share_confirmed": 0.5}
    old = res["gated_agree"]["older"]["CurbRamp"]
    assert old["majority2"]["confirmed"] == 1 and old["margin"]["decided"] == 0
    dup = pd.concat([d.iloc[:1], d.iloc[:1]])
    dup.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not unique"):
        pr.load_assessments(path)


def test_doctests():
    import doctest
    assert doctest.testmod(pr).failed == 0


def test_replay_maps_checks_served_and_gates_per_rule():
    d = pd.DataFrame({"label_type": "CurbRamp", "period": ["round3"] * 3 + ["older"], "ai_result": "Agree",
                      "ai_confidence": [0.6, 0.8, 0.95, 0.95], "ai_accuracy": [0.5, 0.93, 0.97, 0.97],
                      "h_agree": [2, 2, 0, 2], "h_dis": [0, 0, 2, 0]})
    margin = {"correct": {"0.0": 0.5, "0.7": 0.93, "0.9": 0.97}, "incorrect": {"0.0": 0.5}}
    looser = {"correct": {"0.0": 0.95, "0.7": 0.96, "0.9": 0.98}, "incorrect": {"0.0": 0.5}}
    out = pr.replay_maps(d, {"CurbRamp": {"margin": margin, "majority2": looser}})["CurbRamp"]
    assert out["served_check"] == {"n": 3, "share_equal": 1.0}  # the older row is not replayed
    assert out["margin"]["gated"] == 2 and out["margin"]["majority2"]["share_confirmed"] == 0.5
    assert out["majority2"]["gated"] == 3 and out["majority2"]["majority2"]["confirmed"] == 2
