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
    cfp.main(["--assessments", str(src), "--out-dir", str(out), "--holdout-fraction", "0.5", "--seed", "7"])
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
    assert (out / "holdout_bins.csv").exists() and "Held-out" in (out / "summary.md").read_text()
    rep = json.loads((out / "calibration_report.json").read_text())
    assert {c["decision"] for c in rep["classes"]} <= {"clears", "held back", "below gate"}
    assert all(c["fit_range"]["n"] > 0 for c in rep["classes"])


def test_in_sample_is_labelled_and_missing_type_is_tolerated(tmp_path):
    src = tmp_path / "ai_assessments.csv"
    df = _write_assessments(src)
    df[df.label_type != "Obstacle"].to_csv(src, index=False)     # a type with no assessments at all
    out = tmp_path / "phase0"
    cfp.main(["--assessments", str(src), "--out-dir", str(out), "--holdout-fraction", "0",
              "--old-maps-dir", str(_old_maps(tmp_path))])
    md = (out / "summary.md").read_text()
    assert "in-sample" in md and "held-out labelled rows" not in md
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
    # the NoCurbRamp Disagree case on the rounded export: 0.907 observed, SE 0.012 -> held back
    decision, se, lower = cfp.hold_back_decision(True, 0.907, 580)
    assert decision == "held back" and se == pytest.approx(0.012, abs=0.001) and lower < 0.92
    assert cfp.hold_back_decision(True, 0.95, 50)[0] == "held back"   # 0.95 - 1.96*0.031 < 0.92
    assert cfp.hold_back_decision(True, float("nan"), 0)[0] == "held back"


def test_holdout_is_deterministic_and_keyed_on_uid():
    uids = np.array([f"sidewalk_x:{i}" for i in range(4000)])
    a = cfp.holdout_mask(uids, 0.5, 42)
    assert (a == cfp.holdout_mask(uids, 0.5, 42)).all()
    assert 0.45 < a.mean() < 0.55
    assert not cfp.holdout_mask(uids, 0.0, 42).any()


def test_check_old_map_handles_rounded_confidence():
    api_map = {"correct": {"0.5": 0.9, "0.99997": 0.95}, "incorrect": {"0.5": 0.5}}
    rows = pd.DataFrame({"ai_result": ["Agree", "Agree", "Disagree"], "ai_confidence": [1.0, 1.0, 0.7],
                         "ai_accuracy": [0.95, 0.9, 0.5]})  # 2nd: raw 0.99996 -> 1.0
    r = cfp.check_old_map(rows, api_map)
    assert r["exact"] == 2 / 3 and r["consistent"] == 1.0
