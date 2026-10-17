"""End to end on a synthetic assessments file shaped like the production export."""
import json

import numpy as np
import pandas as pd
import pytest

from calibration import _isotonic_map as im
from calibration import calibrate_from_production as cfp


def _write_assessments(path, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(cfp.LABEL_TYPES, n)
    verdict = rng.choice(["Agree", "Disagree"], n, p=[0.8, 0.2])
    conf = np.round(rng.uniform(0.5, 1.0, n), 4)
    # P(humans side with the AI) rises with confidence; Disagree is much less reliable, as in production.
    p_right = np.where(verdict == "Agree", 0.8 + 0.19 * (conf - 0.5) / 0.5, 0.2 + 0.5 * (conf - 0.5) / 0.5)
    right = rng.uniform(size=n) < p_right
    h_n = rng.integers(0, 5, n)
    agree_wins = np.where(verdict == "Agree", right, ~right)
    h_major = np.where(h_n > 0, h_n // 2 + 1, 0)             # strict majority; no votes -> 0 / 0
    h_agree = np.where(agree_wins, h_major, h_n - h_major)
    assert (h_agree >= 0).all() and (h_n - h_agree >= 0).all()
    df = pd.DataFrame({"schema": rng.choice(["sidewalk_a", "sidewalk_b"], n), "label_id": np.arange(n) // 2,
                       "label_type": types, "deleted": "f", "severity": 1, "ai_result": verdict,
                       "ai_accuracy": np.round(np.where(verdict == "Agree", 0.9, 0.95) + 0.04 * (conf - 0.5), 4),
                       "ai_confidence": conf, "ai_date": "2026-01-01",
                       "model_id": ["sidewalk-validator-ai-" + t.lower() for t in types],
                       "h_agree": h_agree, "h_dis": h_n - h_agree, "h_unsure": 0, "jon": None, "mikey": None})
    # make (schema, label_id) unique although label_id repeats across schemas, as in production
    df["schema"] = np.where(np.arange(n) % 2 == 0, "sidewalk_a", "sidewalk_b")
    test_rows = df.head(20).assign(schema="sidewalk_validation", label_id=np.arange(10_000, 10_020))
    pd.concat([df, test_rows]).to_csv(path, index=False)
    return df


def test_end_to_end(tmp_path):
    src = tmp_path / "ai_assessments.csv"
    _write_assessments(src)
    out = tmp_path / "phase0"
    cfp.main(["--assessments", str(src), "--out-dir", str(out), "--folds", "5", "--seed", "7"])
    for t in cfp.LABEL_TYPES:
        api_map = json.loads((out / "accuracy_mappings" / f"{t.lower()}.json").read_text())
        im.validate_api_map(api_map)
        assert (out / f"reliability_{t.lower()}.png").exists()
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 10
    assert summary.n_assessed.sum() == 2000                      # sidewalk_validation rows dropped
    agree = summary[summary.verdict == "Agree"]
    disagree = summary[summary.verdict == "Disagree"]
    assert (agree.observed_precision.values > disagree.observed_precision.values).all()
    # the synthetic old map passes every Disagree verdict; the new one (true precision <= 0.7) should pass few
    assert (disagree.old_coverage == 1.0).all() and disagree.new_coverage.mean() < 0.2
    assert (out / "holdout_bins.csv").exists() and "out-of-fold" in (out / "summary.md").read_text()
    rep = json.loads((out / "calibration_report.json").read_text())
    assert {c["decision"] for c in rep["classes"]} <= {"clears", "held back", "below gate"}
    assert all(c["fit_range"]["n"] > 0 for c in rep["classes"])
    assert rep["outcome"] == "tie_half"
    assert [r["outcome"] for r in rep["sensitivity"]] == ["strict", "share"]
    assert "Sensitivity" in (out / "summary.md").read_text()
    oof = pd.read_csv(out / "labels_oof.csv")
    assert oof[oof.labelled].oof_accuracy.notna().all()


def test_in_sample_is_labelled_and_missing_type_is_tolerated(tmp_path):
    src = tmp_path / "ai_assessments.csv"
    df = _write_assessments(src)
    df[df.label_type != "Obstacle"].to_csv(src, index=False)     # a type with no assessments at all
    out = tmp_path / "phase0"
    cfp.main(["--assessments", str(src), "--out-dir", str(out), "--folds", "1",
              "--old-maps-dir", str(_old_maps(tmp_path))])
    md = (out / "summary.md").read_text()
    assert "in-sample" in md and "out-of-fold" not in md
    obstacle = json.loads((out / "accuracy_mappings" / "obstacle.json").read_text())
    assert set(obstacle["correct"].values()) == {0.0}             # no data -> never clears the gate


def _old_maps(tmp_path):
    d = tmp_path / "old_maps"
    d.mkdir()
    for t in cfp.LABEL_TYPES:
        (d / f"{t.lower()}.json").write_text(json.dumps({"correct": {"0.5": 0.9}, "incorrect": {"0.5": 0.95}}))
    return d


def test_hold_back_rule():
    assert cfp.hold_back_decision(False, 0.99, 1000)[0] == "below gate"
    assert cfp.hold_back_decision(True, 0.99, 1000)[0] == "clears"
    # 0.944 on 126 labels: Wilson 95% lower bound ~0.89 -> held back; at the 0.90 context bound still held back
    decision, lower = cfp.hold_back_decision(True, 0.944, 126)
    assert decision == "held back" and lower == pytest.approx(0.890, abs=0.005)
    assert cfp.hold_back_decision(True, 0.95, 2000)[0] == "clears"            # lower ~0.940
    assert cfp.hold_back_decision(True, 0.93, 2000)[0] == "held back"         # lower ~0.918
    assert cfp.hold_back_decision(True, 0.93, 2000, bound=0.90)[0] == "clears"
    assert cfp.hold_back_decision(True, float("nan"), 0)[0] == "held back"


def test_outcome_definitions():
    df = pd.DataFrame({"ai_result": ["Agree", "Agree", "Disagree", "Agree", "Agree"],
                       "h_agree": [1, 2, 1, 3, 1], "h_dis": [1, 1, 2, 0, 0]})
    df["h_n"] = df.h_agree + df.h_dis
    share = cfp.add_outcomes(df, 2, "share")
    assert share.labelled.tolist() == [True, True, True, True, False]
    assert share.correct[:4].tolist() == pytest.approx([0.5, 2 / 3, 2 / 3, 1.0])
    half = cfp.add_outcomes(df, 2, "tie_half")
    assert half.labelled.tolist() == [True, True, True, True, False]
    assert half.correct[:4].tolist() == [0.5, 1.0, 1.0, 1.0]                 # majority wins count fully
    strict = cfp.add_outcomes(df, 2, "strict")
    assert strict.labelled.tolist() == [False, True, True, True, False]       # the 1-1 tie is dropped
    assert strict.correct[1:4].tolist() == [1.0, 1.0, 1.0]


def test_folds_are_deterministic_and_keyed_on_uid():
    uids = np.array([f"sidewalk_x:{i}" for i in range(5000)])
    f = cfp.fold_ids(uids, 5, 42)
    assert (f == cfp.fold_ids(uids, 5, 42)).all() and set(f) == set(range(5))
    assert np.bincount(f).min() > 900
    assert not cfp.fold_ids(uids, 1, 42).any()
    assert (cfp.fold_ids(uids, 5, 43) != f).mean() > 0.5                   # a different seed reshuffles


def test_cross_fitting_uses_out_of_fold_maps(tmp_path):
    # A class whose high-confidence labels are right only by luck in-sample must not clear on out-of-fold values.
    src = tmp_path / "a.csv"
    df = _write_assessments(src, n=2000)
    res = cfp.calibrate(cfp.load_assessments(src), folds=5, seed=1)
    lab = res["df"][res["df"].labelled]
    assert lab.oof_accuracy.notna().all()
    # out-of-fold values differ from the shipped (all-rows) map's values for some labels
    assert (lab.oof_accuracy - lab.new_accuracy).abs().max() > 0


def test_check_old_map_handles_rounded_confidence():
    api_map = {"correct": {"0.5": 0.9, "0.99997": 0.95}, "incorrect": {"0.5": 0.5}}
    rows = pd.DataFrame({"ai_result": ["Agree", "Agree", "Disagree"], "ai_confidence": [1.0, 1.0, 0.7],
                         "ai_accuracy": [0.95, 0.9, 0.5]})  # 2nd: raw 0.99996 -> 1.0
    r = cfp.check_old_map(rows, api_map)
    assert r["exact"] == 2 / 3 and r["consistent"] == 1.0
