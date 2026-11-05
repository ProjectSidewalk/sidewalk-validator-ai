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


def test_min_n_boundary_and_accuracy_at_exactly_half():
    p = _pred(n=30).assign(p_incorrect=0.5)
    t = sl.slice_table(p, "severity", min_n=15, n_boot=0).set_index("slice")
    assert t.loc["1", "accuracy"] == t.loc["1", "n_incorrect"] / t.loc["1", "n"]   # 0.5 counts as incorrect
    assert np.isnan(sl.slice_table(p, "severity", min_n=16, n_boot=0).accuracy).all()


def test_auc_blank_when_the_smaller_class_is_tiny():
    p = _pred(n=60)
    p.loc[p.index[:30], "true"] = "incorrect"
    p.loc[p.index[0], "true"] = "correct"          # severity-1 slice: 1 correct, 29 incorrect
    t = sl.slice_table(p, "severity", min_n=10, n_boot=0).set_index("slice")
    assert np.isnan(t.loc["1", "auc"]) and not np.isnan(t.loc["1", "accuracy"])
    d = sl.paired_delta_table(p, p, "severity", min_n=10, n_boot=0).set_index("slice")
    assert np.isnan(d.loc["1", "delta"]) and not np.isnan(d.loc["3", "delta"])


def test_bootstrap_groups_by_panorama(monkeypatch):
    seen = []
    real = sl.boot_indices
    monkeypatch.setattr(sl, "boot_indices", lambda y, groups, *a, **k: seen.append(list(groups)) or real(y, groups, *a, **k))
    p = _pred()
    sl.slice_table(p, "severity", min_n=10, n_boot=5)
    assert seen and set(seen[0]) <= set(p.pano_id)
    idx = real(np.array([0, 1, 0, 1], bool), np.array(["a", "a", "b", "b"]), 20, np.random.default_rng(0))
    assert all(sorted(i) in ([0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 2, 3], [2, 2, 3, 3], [2, 3, 2, 3])
               or (len(i) == 4 and all(np.sum(i == k) == np.sum(i == k ^ 1) for k in (0, 2))) for i in map(np.array, idx))


def test_bootstrap_interval_is_the_2_5_and_97_5_percentiles():
    p = _pred()
    g = p[p.severity == 3]
    y = (g["true"] == "incorrect").to_numpy()
    lo, hi = sl.bootstrap_auc(y, g.p_incorrect, g.pano_id, 300, seed=7)
    idxs = sl.boot_indices(y, g.pano_id, 300, np.random.default_rng(7))
    vals = [sl._auc(y[i], g.p_incorrect.to_numpy()[i]) for i in idxs]
    assert (lo, hi) == (pytest.approx(np.percentile(vals, 2.5)), pytest.approx(np.percentile(vals, 97.5)))


def test_paired_delta_uses_the_same_resample_for_both_models():
    a = _pred()
    b = a.assign(p_incorrect=a.p_incorrect + np.random.default_rng(1).normal(0, 1e-4, len(a)))
    t = sl.paired_delta_table(a, b, "severity", min_n=10, n_boot=300).set_index("slice")
    assert (t.loc["3", "delta_hi"] - t.loc["3", "delta_lo"]) < 0.02   # unpaired resamples would give ~0.3 wide


def test_paired_delta_refuses_different_label_sets():
    a = _pred()
    with pytest.raises(ValueError, match="no score for 1"):
        sl.paired_delta_table(a, a.iloc[1:], "tag")
    extra = pd.concat([a, a.assign(label_uid="x" + a.label_uid)])     # the baseline may score more rows
    assert len(sl.paired_delta_table(a, extra, "tag", n_boot=0)) == 2


def test_thin_flag_boundary():
    p = _pred(n=2).iloc[:1]
    train = pd.concat([p] * sl.THIN_TRAIN)
    assert not sl.slice_table(p, "tag", train=train, n_boot=0).needs_data.iloc[0]
    assert sl.slice_table(p, "tag", train=train.iloc[1:], n_boot=0).needs_data.iloc[0]


def test_slice_report_strict_json_and_severity_order_and_city_share():
    p = _pred().assign(city=["a"] * 50 + ["b"] * 10)
    json.dumps(sl.slice_report(p, min_n=31, n_boot=0), allow_nan=False)   # every rate blank: None, not NaN
    r = sl.slice_report(p, min_n=10, n_boot=10)
    json.dumps(r, allow_nan=False)
    sev = [row["slice"] for row in r["test"]["severity"]]
    assert sev == ["1", "3"]
    share = {row["slice"]: row["max_city_share"] for row in r["test"]["severity"]}
    assert share == {"1": 1.0, "3": pytest.approx(20 / 30)}


def test_retrain_report_wiring_train_split_boot_and_both_splits(tmp_path):
    from types import SimpleNamespace
    from analysis import retrain_report as rr
    p = pd.concat([_pred(), _pred().assign(split="test_heldout", label_uid=lambda d: "h" + d.label_uid,
                                            pano_id=lambda d: "h" + d.pano_id)])
    split_all = pd.concat([p, _pred().iloc[:20].assign(split="train", label_uid=lambda d: "t" + d.label_uid)])
    (tmp_path / "CurbRamp").mkdir()
    p[["label_uid", "p_incorrect"]].to_csv(tmp_path / "CurbRamp" / "predictions_prod.csv", index=False)
    a, b = rr.slice_rows(str(tmp_path), "CurbRamp", p, split_all, SimpleNamespace(slice_min_n=10, n_boot=30, seed=0))
    sa = pd.DataFrame(a)
    assert set(sa.split) == {"test", "test_heldout"}
    assert sa[(sa.by == "tag") & (sa.slice == "steep")].train_n.unique().tolist() == [20]
    assert sa.auc_lo.notna().any()
    a0, _ = rr.slice_rows(str(tmp_path), "CurbRamp", p, split_all, SimpleNamespace(slice_min_n=10, n_boot=0, seed=0))
    assert pd.DataFrame(a0).auc_lo.isna().all()


def test_hf_card_and_publish_drop_slices():
    import hf_cards
    import publish_to_hf
    m = {"test": {"n": 2, "n_by_class": {"correct": 1, "incorrect": 1}, "accuracy": 1.0, "balanced_accuracy": 1.0,
                  "auc": 1.0, "per_class": {}}, "slices": {"test": {"tag": []}}}
    assert "| slices |" not in hf_cards._metrics_section(m)
    assert "slices" not in publish_to_hf.pool_private_city_metrics(m, ".", "CurbRamp", set())


def test_auc_needs_eight_of_each_class():
    base = _pred(n=40).iloc[:0]
    rows = pd.DataFrame({"label_uid": [f"x:{i}" for i in range(40)], "pano_id": [f"p{i}" for i in range(40)],
                         "true": ["incorrect"] * 8 + ["correct"] * 32, "p_incorrect": np.linspace(1, 0, 40),
                         "severity": 1, "tags": "[]"})
    assert sl.slice_table(rows, "severity", min_n=10, n_boot=0).auc.iloc[0] == 1.0
    seven = rows.assign(true=["incorrect"] * 7 + ["correct"] * 33)
    assert np.isnan(sl.slice_table(seven, "severity", min_n=10, n_boot=0).auc.iloc[0])
    assert np.isnan(sl.bootstrap_auc((seven.true == "incorrect").to_numpy(), seven.p_incorrect, seven.pano_id, 10)[0])
    assert not np.isnan(sl.bootstrap_auc((rows.true == "incorrect").to_numpy(), rows.p_incorrect, rows.pano_id, 10)[0])
    assert np.isnan(sl.paired_delta_table(seven, seven, "severity", min_n=10, n_boot=0).delta.iloc[0])
    assert not np.isnan(sl.paired_delta_table(rows, rows, "severity", min_n=10, n_boot=0).delta.iloc[0])


def test_severity_none_sorts_last_and_paired_carries_city_share():
    p = _pred().assign(city="a")
    p.loc[p.index[:5], "severity"] = np.nan
    assert sl.slice_table(p, "severity", min_n=1, n_boot=0).slice.tolist() == ["1", "3", "(none)"]
    d = sl.paired_delta_table(p, p, "severity", min_n=1, n_boot=0)
    assert d.slice.tolist() == ["1", "3", "(none)"] and (d.max_city_share == 1.0).all()


def test_per_slice_seeds_keep_an_interval_stable_when_other_slices_change():
    p = _pred()
    t1 = sl.slice_table(p, "tag", min_n=10, n_boot=100).set_index("slice")
    extra = p.iloc[:12].assign(label_uid=lambda d: "e" + d.label_uid, tags='["new"]')
    t2 = sl.slice_table(pd.concat([p, extra]), "tag", min_n=10, n_boot=100).set_index("slice")
    assert t1.loc["narrow", ["auc_lo", "auc_hi"]].tolist() == t2.loc["narrow", ["auc_lo", "auc_hi"]].tolist()
    d1 = sl.paired_delta_table(p, p.assign(p_incorrect=1 - p.p_incorrect), "tag", min_n=10, n_boot=100)
    d2 = sl.paired_delta_table(pd.concat([p, extra]), pd.concat([p, extra]).assign(p_incorrect=lambda d: 1 - d.p_incorrect),
                               "tag", min_n=10, n_boot=100)
    assert d1.set_index("slice").loc["narrow", "delta_lo"] == d2.set_index("slice").loc["narrow", "delta_lo"]


def test_boot_indices_redraws_single_class_resamples():
    y = np.array([True] + [False] * 9)
    out = sl.boot_indices(y, np.arange(10), 50, np.random.default_rng(0), max_tries=50)
    assert len(out) == 50 and all(0 < y[i].sum() < len(i) for i in out)
    with pytest.warns(UserWarning, match="had one class"):
        assert len(sl.boot_indices(y, np.arange(10), 50, np.random.default_rng(0), max_tries=1)) < 50


def test_paired_delta_groups_by_panorama_and_checks_true(monkeypatch):
    seen = []
    real = sl.boot_indices
    monkeypatch.setattr(sl, "boot_indices", lambda y, groups, *a, **k: seen.append(set(groups)) or real(y, groups, *a, **k))
    p = _pred()
    sl.paired_delta_table(p, p, "severity", min_n=10, n_boot=5)
    assert seen and seen[0] <= set(p.pano_id)
    with pytest.raises(ValueError, match="true class"):
        sl.paired_delta_table(p, p.assign(true=p["true"][::-1].to_numpy()), "severity", n_boot=0)


def test_evaluate_slices_carry_city_share_pano_bootstrap_and_default_min_n(tmp_path, monkeypatch):
    import evaluate
    seen = []
    real = sl.boot_indices
    monkeypatch.setattr(sl, "boot_indices", lambda y, groups, *a, **k: seen.append(set(groups)) or real(y, groups, *a, **k))
    p = _pred()
    lt = tmp_path / "CurbRamp"
    lt.mkdir()
    split = p.assign(city=["a"] * 40 + ["b"] * 20, label_id=range(len(p)), label_type="CurbRamp",
                     label_class=p["true"], crop_path="x.webp", hard=0)
    split.to_csv(lt / "split.csv", index=False)
    p.assign(p_correct=1 - p.p_incorrect)[evaluate.PREDICTION_COLUMNS].to_csv(lt / "predictions.csv", index=False)
    evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference", "--slice-boot", "5"])
    rows = {r["slice"]: r for r in json.load(open(lt / "metrics.json"))["slices"]["test"]["severity"]}
    assert rows["3"]["max_city_share"] == pytest.approx(20 / 30)
    assert seen and all(g <= set(p.pano_id) for g in seen)
    assert f"blank under {sl.MIN_N} " in (lt / "slices.md").read_text()
