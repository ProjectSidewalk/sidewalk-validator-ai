"""Tests for analysis/slices.py (performance by label type x severity and x tag) and its use in evaluate.py."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest

from analysis import slices as sl


def _pred(n=60, seed=0):
    """Predictions with a perfectly ranked half ('steep', severity 1) and a random half ('narrow', severity 3)."""
    rng = np.random.default_rng(seed)
    y = np.array(["correct", "incorrect"] * (n // 2))
    good = (y == "incorrect") * 0.8 + 0.1
    bad = rng.random(n)
    half = np.arange(n) < n // 2
    return pd.DataFrame({"label_uid": [f"c:{i}" for i in range(n)], "pano_id": [f"p{i // 2}" for i in range(n)],
                         "split": "test", "true": y, "p_incorrect": np.where(half, good, bad),
                         "severity": np.where(half, 1, 3), "tags": np.where(half, '["steep"]', '["narrow"]')})


def test_doctests():
    assert doctest.testmod(sl).failed == 0


def test_parse_tags_encodings():
    assert sl.parse_tags('["a b","c"]') == ["a b", "c"]
    assert sl.parse_tags("a b;c") == ["a b", "c"]
    assert sl.parse_tags(np.nan) == [] and sl.parse_tags("") == [] and sl.parse_tags(["x", ""]) == ["x"]


def test_with_slice_severity_none_and_tag_explode():
    d = pd.DataFrame({"severity": [1.0, np.nan, 3], "tags": ['["a","b"]', "[]", None]})
    assert list(sl.with_slice(d, "severity").slice) == ["1", "(none)", "3"]
    assert sorted(sl.with_slice(d, "tag").slice) == ["(none)", "(none)", "a", "b"]
    with pytest.raises(ValueError):
        sl.with_slice(d, "city")


def test_slice_table_separates_good_and_bad_slices_and_flags_thin_training():
    p = _pred()
    train = p.iloc[:40]          # 30 steep, 10 narrow training rows
    t = sl.slice_table(p, "tag", train=train, min_n=10, n_boot=200).set_index("slice")
    assert t.loc["steep", "auc"] == 1.0 and t.loc["narrow", "auc"] < 0.8
    assert t.loc["steep", "auc_lo"] <= 1.0 <= t.loc["steep", "auc_hi"] + 1e-12
    assert (t.loc["steep", "train_n"], t.loc["narrow", "train_n"]) == (30, 10)
    assert bool(t.loc["narrow", "needs_data"]) and bool(t.loc["steep", "needs_data"])  # both under THIN_TRAIN=100


def test_slice_table_blanks_small_slices_but_counts_them():
    t = sl.slice_table(_pred(), "severity", min_n=31, n_boot=0)
    assert t.n.tolist() == [30, 30] and t.auc.isna().all() and t.accuracy.isna().all()


def test_bootstrap_needs_enough_of_each_class_and_resamples_panoramas():
    y = np.array([1] * 5 + [0] * 20, bool)
    assert np.isnan(sl.bootstrap_auc(y, np.linspace(0, 1, 25), np.arange(25))[0])
    y = np.array([0, 1] * 20, bool)
    s = y * 0.8 + 0.1
    lo, hi = sl.bootstrap_auc(y, s, np.repeat(np.arange(20), 2), n_boot=100)
    assert lo == hi == 1.0


def test_paired_delta_same_rows_and_sign():
    a = _pred()
    b = a.assign(p_incorrect=np.where(a.tags == '["steep"]', 1 - a.p_incorrect, a.p_incorrect))
    t = sl.paired_delta_table(a, b, "tag", min_n=10, n_boot=200).set_index("slice")
    assert t.loc["steep", "delta"] == pytest.approx(1.0)     # A perfect, B perfectly inverted
    assert t.loc["steep", "delta_lo"] > 0
    assert t.loc["narrow", "delta"] == pytest.approx(0.0)
    with pytest.raises(pd.errors.MergeError):
        sl.paired_delta_table(a, pd.concat([b, b]), "tag")


def test_slice_report_is_json_serialisable_and_markdown_renders():
    p = pd.concat([_pred(), _pred().assign(split="test_heldout", label_uid=lambda d: "h" + d.label_uid)])
    r = sl.slice_report(p, train=p.iloc[:10], min_n=10, n_boot=50)
    assert set(r) == {"test", "test_heldout"} and set(r["test"]) == {"severity", "tag"}
    json.dumps(r)                                            # no NaN objects, no numpy scalars that json rejects
    md = sl.report_markdown(r, "CurbRamp")
    assert "### test, by tag" in md and "| steep |" in md and "⚠ thin" in md


def test_evaluate_writes_slices(tmp_path):
    import evaluate
    p = _pred()
    lt = tmp_path / "CurbRamp"
    lt.mkdir()
    split = p.assign(city="c", label_id=range(len(p)), label_type="CurbRamp", label_class=p["true"],
                     crop_path="x.webp", hard=0)
    train = split.iloc[:20].assign(split="train", label_uid=lambda d: "t" + d.label_uid)
    pd.concat([split, train]).to_csv(lt / "split.csv", index=False)
    p.assign(p_correct=1 - p.p_incorrect)[evaluate.PREDICTION_COLUMNS].to_csv(lt / "predictions.csv", index=False)
    evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference",
                   "--min-slice-n", "10", "--slice-boot", "20"])
    m = json.load(open(lt / "metrics.json"))
    rows = {r["slice"]: r for r in m["slices"]["test"]["tag"]}
    assert rows["steep"]["auc"] == 1.0 and rows["steep"]["train_n"] == 20 and rows["narrow"]["train_n"] == 0
    assert (lt / "slices.md").exists()


def test_retrain_report_slice_rows_and_section(tmp_path):
    from types import SimpleNamespace
    from analysis import retrain_report as rr
    p = _pred()
    (tmp_path / "CurbRamp").mkdir()
    split_all = pd.concat([p, p.iloc[:20].assign(split="train", label_uid=lambda d: "t" + d.label_uid)])
    p.assign(p_incorrect=1 - p.p_incorrect)[["label_uid", "p_incorrect"]].to_csv(
        tmp_path / "CurbRamp" / "predictions_prod.csv", index=False)
    args = SimpleNamespace(slice_min_n=10, n_boot=50, seed=0)
    a, b = rr.slice_rows(str(tmp_path), "CurbRamp", p, split_all, args)
    sa, sb = pd.DataFrame(a), pd.DataFrame(b)
    assert set(sa.by) == {"severity", "tag"} and (sa.split == "test").all()
    steep = sb[(sb.by == "tag") & (sb.slice == "steep")].iloc[0]
    assert steep.auc_a == 1.0 and steep.auc_b == 0.0 and steep.delta == pytest.approx(1.0)
    text = "\n".join(rr.slice_section(sa, sb))
    assert "CurbRamp / test / by tag: round 3 vs production" in text
    a2, b2 = rr.slice_rows(str(tmp_path / "nowhere"), "CurbRamp", p, split_all, args)
    assert a2 and not b2                                    # no predictions_prod.csv: slices only, no comparison
