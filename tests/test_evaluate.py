"""Metric functions in evaluate.py (issue #6), on inputs small enough to check by hand."""
import json

import numpy as np
import pandas as pd
import pytest

from evaluate import format_report, recall_at_precision, report

# Scores sorted high to low; positives at ranks 1, 2 and 4. Operating points (threshold: precision, recall):
#   0.4: 3/6, 1    0.55: 3/5, 1    0.6: 3/4, 1    0.7: 2/3, 2/3    0.8: 2/2, 2/3    0.9: 1/1, 1/3
# The precision array is not monotone (0.75 then 0.667), which is what broke searchsorted. Among thresholds with
# the same best recall, the highest is reported.
Y = [1, 1, 0, 1, 0, 0]
S = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4]


@pytest.mark.parametrize("level, recall, threshold", [
    (0.50, 1.0, 0.6),     # recall 1 at thresholds 0.4, 0.55 and 0.6: the highest is reported
    (0.60, 1.0, 0.6),
    (0.70, 1.0, 0.6),
    (0.75, 1.0, 0.6),
    (0.90, 2 / 3, 0.8),
    (1.00, 2 / 3, 0.8),
])
def test_recall_at_precision_hand_computed(level, recall, threshold):
    r, t = recall_at_precision(Y, S, level)
    assert r == pytest.approx(recall)
    assert t == pytest.approx(threshold)


def test_recall_at_precision_unreachable_level():
    # Only the terminal (precision 1, recall 0) point has precision >= 1 here, and it has no threshold.
    r, t = recall_at_precision([1, 0, 1, 0], [0.9, 0.95, 0.5, 0.6], 1.0)
    assert (r, t) == (0.0, None)


def _synthetic_predictions():
    # test: 4 correct, 4 incorrect. Scores chosen so the AUC is 15/16 and the argmax confusion is known.
    true = ["correct"] * 4 + ["incorrect"] * 4
    p_inc = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
    rows = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(8)], "split": "test", "true": true,
                         "p_incorrect": p_inc})
    held = pd.DataFrame({"label_uid": ["h:1", "h:2"], "split": "test_heldout", "true": ["correct", "incorrect"],
                         "p_incorrect": [0.2, 0.9]})
    val = pd.DataFrame({"label_uid": ["v:1"], "split": "val", "true": ["correct"], "p_incorrect": [0.99]})
    df = pd.concat([rows, held, val], ignore_index=True)
    df["p_correct"] = 1 - df.p_incorrect
    df["hard"] = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    return df


def test_report_known_values():
    m = report(_synthetic_predictions())
    assert set(m) == {"test", "test_heldout", "hard"}
    t = m["test"]
    # pairs (incorrect, correct) ranked right: 0.4 beats 3 of 4 correct, the rest beat all 4 -> 15/16
    assert t["auc"] == pytest.approx(15 / 16)
    # argmax: correct-row 0.6 -> predicted incorrect; incorrect-row 0.4 -> predicted correct
    assert t["accuracy"] == pytest.approx(6 / 8)
    assert t["balanced_accuracy"] == pytest.approx(0.75)
    assert t["per_class"]["incorrect"]["precision"] == pytest.approx(3 / 4)
    assert t["per_class"]["incorrect"]["recall"] == pytest.approx(3 / 4)
    assert t["confusion_matrix"] == [[3, 1], [1, 3]]
    assert t["recall_at_precision"]["incorrect"]["0.90"] == {"recall": pytest.approx(0.75), "threshold": pytest.approx(0.7)}
    assert m["test_heldout"]["accuracy"] == 1.0
    # hard slice: test rows 0, 3, 4 (val row 10 is excluded)
    assert m["hard"]["n"] == 3
    assert "accuracy" in format_report(m) and "balanced accuracy" in format_report(m)


def test_report_without_heldout_or_hard():
    df = _synthetic_predictions().drop(columns="hard")
    df = df[df.split != "test_heldout"]
    assert set(report(df)) == {"test"}


@pytest.mark.parametrize("argv, message", [
    (["--checkpoint", "projectsidewalk/sidewalk-validator-ai-curbramp"], "needs --tag"),
    (["--revision", "abc"], "applies only to --checkpoint"),
])
def test_comparison_model_flags_are_checked_before_any_work(argv, message, capsys):
    from evaluate import main
    with pytest.raises(SystemExit):
        main(["--label_type", "CurbRamp", "--data-dir", "/nonexistent", *argv])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv, message", [
    (["--checkpoint", "m", "--tag", "prod", "--skip-inference"], "would be ignored"),
    (["--tag", "../prod"], "--tag must be"),
])
def test_more_flag_checks(argv, message, capsys):
    from evaluate import main
    with pytest.raises(SystemExit):
        main(["--label_type", "CurbRamp", "--data-dir", "/nonexistent", *argv])
    assert message in capsys.readouterr().err


def test_skip_inference_with_a_tag_writes_only_suffixed_outputs_and_no_hint(tmp_path, capsys):
    from evaluate import main
    base = tmp_path / "CurbRamp"
    base.mkdir()
    rows = []
    for i in range(40):
        cls = "correct" if i % 2 else "incorrect"
        rows.append({"label_uid": f"seattle-wa:{i}", "city": "seattle-wa", "label_id": i, "pano_id": f"p{i}",
                     "label_type": "CurbRamp", "label_class": cls, "crop_path": f"c{i}.webp",
                     "split": "test" if i < 30 else "val", "hard": 0})
    split = pd.DataFrame(rows)
    split.to_csv(base / "split.csv", index=False)
    rng = np.random.default_rng(0)
    p_inc = np.clip((split.label_class == "incorrect") * 0.6 + rng.uniform(0, 0.4, len(split)), 0, 1)
    pd.DataFrame({"label_uid": split.label_uid, "split": split.split, "true": split.label_class,
                  "p_correct": 1 - p_inc, "p_incorrect": p_inc}).to_csv(base / "predictions_prod.csv", index=False)
    main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference", "--tag", "prod"])
    written = sorted(f.name for f in base.iterdir())
    assert written == ["metrics_prod.json", "predictions_prod.csv", "pvr_prod.png", "report_prod.png",
                       "report_prod.txt", "split.csv"]
    assert "calibration.calibrate" not in capsys.readouterr().out


# --- per-city breakdown (issue #15) -------------------------------------------------------------------------------

def _per_city_predictions():
    # test: city a has 6 rows (both classes), city b 4 rows (correct only), city c 1 row; heldout: city d, 3 rows.
    df = pd.DataFrame({
        "label_uid": [f"x:{i}" for i in range(14)],
        "split": ["test"] * 11 + ["test_heldout"] * 3,
        "city": ["a"] * 6 + ["b"] * 4 + ["c"] + ["d"] * 3,
        "true": ["correct", "correct", "correct", "incorrect", "incorrect", "incorrect",
                 "correct", "correct", "correct", "correct", "incorrect", "correct", "incorrect", "incorrect"],
        "p_incorrect": [0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.1, 0.2, 0.3, 0.6, 0.9, 0.2, 0.8, 0.4]})
    df["p_correct"] = 1 - df.p_incorrect
    return df


def test_per_city_slices_use_slice_metrics_and_respect_min_n():
    df = _per_city_predictions()
    m = report(df, per_city=True, min_city_n=3)
    assert set(m) == {"test", "test_heldout", "per_city"}
    pc = m["per_city"]["test"]
    assert list(pc["cities"]) == ["a", "b"]          # by descending n; c has 1 row < 3
    assert pc["below_min_n"] == {"c": 1}
    a = pc["cities"]["a"]
    # a: verdicts c, c, i, i, i, c -> 4 of 6 right; correct precision 2/3, incorrect recall 2/3
    assert a["accuracy"] == pytest.approx(4 / 6)
    assert a["per_class"]["correct"]["precision"] == pytest.approx(2 / 3)
    assert a["per_class"]["incorrect"]["recall"] == pytest.approx(2 / 3)
    from evaluate import slice_metrics
    assert a == slice_metrics(df[(df.split == "test") & (df.city == "a")])
    assert pc["cities"]["b"]["auc"] is None          # one class only
    assert list(m["per_city"]["test_heldout"]["cities"]) == ["d"]


def test_per_city_is_absent_by_default_and_skipped_by_format_report():
    from evaluate import format_per_city
    df = _per_city_predictions()
    assert "per_city" not in report(df)
    m = report(df, per_city=True, min_city_n=3)
    assert format_report(m) == format_report(report(df))
    md = format_per_city(m["per_city"])
    assert "### test: per city (n >= 3)" in md and "| a | 6 (3 / 3) | 0.667 |" in md
    assert "| b | 4 (4 / 0) | 0.750 | — |" in md and "c (n=1)" in md


def test_per_city_needs_a_city_column():
    with pytest.raises(ValueError, match="city"):
        report(_per_city_predictions().drop(columns="city"), per_city=True)


def test_main_per_city_writes_markdown_and_json(tmp_path):
    from evaluate import main
    base = tmp_path / "CurbRamp"
    base.mkdir()
    preds = _per_city_predictions()
    split = pd.DataFrame({"label_uid": preds.label_uid, "city": preds.city, "label_id": range(len(preds)),
                          "pano_id": [f"p{i}" for i in range(len(preds))], "label_type": "CurbRamp",
                          "label_class": preds["true"], "crop_path": "c.webp", "split": preds.split, "hard": 0})
    split.to_csv(base / "split.csv", index=False)
    preds.drop(columns="city").to_csv(base / "predictions.csv", index=False)
    main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference", "--per-city",
          "--min-city-n", "3"])
    metrics = json.loads((base / "metrics.json").read_text())
    assert list(metrics["per_city"]["test"]["cities"]) == ["a", "b"]
    assert (base / "per_city.md").read_text().startswith("### test: per city")


def test_model_card_ignores_per_city():
    from hf_cards import _metrics_section
    m = report(_per_city_predictions(), per_city=True, min_city_n=3)
    assert "per_city" not in _metrics_section(m)
