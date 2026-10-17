"""Prior reweighting in calibration/calibrate.py, checked against Bayes' rule in closed form."""
import json

import numpy as np
import pandas as pd
import pytest

from calibration import _isotonic_map as im
from calibration import calibrate as cal


def _predictions(n_correct, n_incorrect, acc_correct, acc_incorrect, split, conf=0.8):
    """Exact class-conditional accuracies: a fraction acc_y of class-y rows is predicted as y, the rest flipped.

    Every row has the same confidence, so the isotonic fit is one block and equals the weighted precision.
    """
    rows = []
    for true, n, acc in (("correct", n_correct, acc_correct), ("incorrect", n_incorrect, acc_incorrect)):
        k = int(round(n * acc))
        for i in range(n):
            pred = true if i < k else ("incorrect" if true == "correct" else "correct")
            p_inc = conf if pred == "incorrect" else 1 - conf
            rows.append({"label_uid": f"{split}:{true}:{i}", "split": split, "true": true,
                         "p_correct": 1 - p_inc, "p_incorrect": p_inc})
    return pd.DataFrame(rows)


def test_prior_reweighting_matches_bayes(tmp_path):
    a, b = 0.9, 0.8                                   # P(say correct | correct), P(say incorrect | incorrect)
    preds = pd.concat([_predictions(5500, 4500, a, b, "val"), _predictions(5500, 4500, a, b, "test")])
    pi = {"correct": 0.9, "incorrect": 0.1}
    api_map, rep, _ = cal.calibrate(preds, pi)
    want_correct = pi["correct"] * a / (pi["correct"] * a + pi["incorrect"] * (1 - b))      # 0.81 / 0.83
    want_incorrect = pi["incorrect"] * b / (pi["incorrect"] * b + pi["correct"] * (1 - a))  # 0.08 / 0.17
    curves, _ = cal.fit_curves(cal.verdicts(preds[preds.split == "val"]), pi)
    assert float(curves["correct"].value(0.8)[0]) == pytest.approx(want_correct, abs=1e-9)
    assert float(curves["incorrect"].value(0.8)[0]) == pytest.approx(want_incorrect, abs=1e-9)
    # blocks of thousands are served at their mean (the thin-block guard does not apply)
    assert im.api_lookup(api_map["correct"], 0.8) == pytest.approx(want_correct, abs=1e-4)
    assert im.api_lookup(api_map["incorrect"], 0.8) == pytest.approx(want_incorrect, abs=1e-4)
    # Without reweighting (55:45) the incorrect verdict would look far better than it is in deployment.
    unweighted = 0.45 * b / (0.45 * b + 0.55 * (1 - a))
    assert unweighted > 0.85 > want_incorrect
    assert rep["eval"]["incorrect"]["precision"] == pytest.approx(want_incorrect, abs=1e-9)
    assert rep["eval"]["correct"]["gate_coverage"] == 1.0 and rep["eval"]["incorrect"]["gate_coverage"] == 0.0


def test_prior_weights_and_load_prior(tmp_path):
    w = cal.prior_weights(["correct"] * 3 + ["incorrect"], {"correct": 0.5, "incorrect": 0.5})
    assert np.allclose(w, [0.5 / 0.75] * 3 + [0.5 / 0.25])
    p = tmp_path / "deployment_priors.json"
    p.write_text(json.dumps({"CurbRamp": {"correct": 3969, "incorrect": 417, "measured_on": "2026-10-17",
                                          "source": "test"}}))
    pi, entry = cal.load_prior(p, "CurbRamp")
    assert pi["correct"] == pytest.approx(3969 / 4386) and entry["source"] == "test"
    with pytest.raises(KeyError):
        cal.load_prior(p, "Obstacle")


def test_cli_writes_map_and_report(tmp_path):
    preds = pd.concat([_predictions(550, 450, 0.9, 0.8, "val"), _predictions(550, 450, 0.9, 0.8, "test")])
    preds.to_csv(tmp_path / "predictions.csv", index=False)
    (tmp_path / "priors.json").write_text(json.dumps({"Obstacle": {"correct": 9, "incorrect": 1,
                                                                   "measured_on": "x", "source": "y"}}))
    out = tmp_path / "cal" / "obstacle.json"
    cal.main(["--predictions", str(tmp_path / "predictions.csv"), "--priors", str(tmp_path / "priors.json"),
              "--label-type", "Obstacle", "--out", str(out)])
    im.validate_api_map(json.loads(out.read_text()))
    rep = json.loads((tmp_path / "cal" / "calibration_report.json").read_text())
    assert rep["label_type"] == "Obstacle" and "overall_ece" in rep["eval"]
    assert (tmp_path / "cal" / "reliability.png").exists()


def test_eval_split_missing_a_class_still_writes_map(tmp_path):
    preds = pd.concat([_predictions(550, 450, 0.9, 0.8, "val"), _predictions(100, 0, 0.9, 0.8, "test")])
    with pytest.warns(UserWarning):
        api_map, rep, tables = cal.calibrate(preds, {"correct": 0.9, "incorrect": 0.1})
    im.validate_api_map(api_map)
    assert rep["eval"] == "n/a" and tables == {}
    assert rep["fit_range"]["correct"]["n"] > 0
