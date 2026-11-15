"""Tests for analysis/vlm_stacking.py (the Flash + round-3 stacking check). CPU only, synthetic scores, no files."""
import doctest

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from analysis import vlm_stacking as vs  # noqa: E402


def frame(n_cities=8, per=40, seed=0):
    """Synthetic crops of every label type: two noisy scores of the same truth, with independent noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in vs.LABEL_TYPES:
        for c in range(n_cities):
            y = rng.integers(0, 2, per)
            rows.append(pd.DataFrame({
                "city": f"city-{c}", "label_id": np.arange(per) + 1000 * vs.LABEL_TYPES.index(t),
                "label_type": t, "y": y, "pano_id": [f"p{c}-{t}-{i // 2}" for i in range(per)],
                "p_r3": 1 / (1 + np.exp(-(2 * (2 * y - 1) + rng.normal(0, 2, per)))),
                "p_vlm": np.round(1 / (1 + np.exp(-(2 * (2 * y - 1) + rng.normal(0, 2, per)))), 2)}))
    df = pd.concat(rows, ignore_index=True)
    # the same label_id in every city, as in Project Sidewalk: only (city, label_id) is a key
    return df.assign(label_uid=df.city + ":" + df.label_id.astype(str))


def test_doctests():
    res = doctest.testmod(vs)
    assert res.failed == 0 and res.attempted > 0


def test_uid_is_city_and_label_id():
    df = frame()
    assert df.label_id.duplicated().any() and vs.with_uid(df).label_uid.is_unique
    with pytest.raises(ValueError, match="duplicate"):
        vs.with_uid(pd.concat([df, df.head(1)]))
    with pytest.raises(ValueError, match="not city:label_id"):
        vs.with_uid(df.assign(label_uid=df.label_id.astype(str)))


def test_folds_never_split_a_city(monkeypatch):
    df, seen = frame(), []
    real = vs._fit_predict

    def spy(xtr, ytr, xte, raw=False):
        seen.append((len(xtr), len(xte)))
        return real(xtr, ytr, xte, raw)
    monkeypatch.setattr(vs, "_fit_predict", spy)
    g = df[df.label_type == vs.LABEL_TYPES[0]]
    x = np.column_stack([vs.logit(g.p_r3, vs.CLIP["p_r3"]), vs.logit(g.p_vlm, vs.CLIP["p_vlm"])])
    scores, w = vs.oof_scores(x, g.y.to_numpy(), g.city.to_numpy())
    assert len(seen) == vs.N_FOLDS and all(a + b == len(g) and b % 40 == 0 for a, b in seen)   # whole cities
    assert w.shape == (vs.N_FOLDS, 1) and not np.isnan(scores).any()


def test_stack_gains_when_the_second_score_is_independent_and_holdout_flags_trained_rows():
    df = frame()
    res = vs.analyse(df, n_boot=100)
    yes, row = vs.verdict(res["paired"])
    assert yes and row["diff"] > 0.02 and row.city_lo > 0
    a = res["auc"].set_index(["label_type", "model"]).auc
    assert a["macro", "stack_type"] > a["macro", "round3"] > 0.5
    run = df[["label_uid", "city", "label_id", "label_type", "pano_id"]].assign(split="test")
    run.loc[0, "split"] = "train"                       # one crop trained on; its panorama mate is flagged too
    other = df[df.label_type != df.label_type[0]].iloc[:1]   # a train row of another type on crop 0's panorama
    run = pd.concat([run, other.assign(label_id=999999, label_uid=other.city + ":999999",
                                       pano_id=df.pano_id[0], split="train")[run.columns]], ignore_index=True)
    rows, trained = vs.holdout_check(df, run)
    n = rows.set_index("check").n
    assert trained.sum() == 2 and n["round-3 split = train"] == 1
    assert n[[c for c in n.index if "different label type" in c][0]] == 2   # crop 0 and its panorama mate
    assert "pre-stated rule: YES" in vs.render({"n_both": len(df), "n_vlm": len(df), "n_classifier": len(df),
                                        "vlm_unscored": 0}, rows, res, None, 0)


def test_round3_clip_keeps_its_tails_distinct():
    """The primary clip must not tie round 3's stored scores (about 1e-11 to 1 - 1e-7, plus exact 1.0)."""
    x = vs.logit(np.array([1.2e-11, 1e-9, 1e-7, 1 - 1e-7, 1.0]), vs.CLIP["p_r3"])
    assert np.isfinite(x).all() and (np.diff(x) > 0).all()


def test_error_rows_expectation_under_independence():
    # one type; round 3 wrong on crops 0 and 2 (one of each class); Flash's accuracy is 1/2 on class 1, 1 on class 0
    df = pd.DataFrame({"label_type": vs.LABEL_TYPES[0], "y": [1, 1, 0, 0],
                       "p_r3": [0.1, 0.9, 0.9, 0.1], "p_vlm": [0.9, 0.1, 0.1, 0.1]})
    r = vs.error_rows(df).iloc[0]
    assert r.n_round3_wrong == 2 and r.flash_right_when_round3_wrong == 1.0
    assert r.flash_right_if_independent == pytest.approx((0.5 + 1.0) / 2)
    assert (r.flash_tpr, r.flash_tnr) == (0.5, 1.0)


def test_sensitivity_is_reported_apart_from_the_primary():
    df = frame()
    res = vs.analyse(df, n_boot=50)
    sens, seeds = vs.sensitivity(df, n_boot=50, fold_seeds=(0, 1), n_refit=5)
    assert list(sens.variant) == ["clip_1e-6", "normal_rank", "loco", "refit_bootstrap"]
    assert list(seeds.fold_seed) == [0, 1] and seeds["diff"].notna().all()
    assert (sens.set_index("variant").x0_alone_diff[["normal_rank", "loco"]] == 0).all()   # same ranking as round 3
    page = vs.render({"n_both": len(df), "n_vlm": len(df), "n_classifier": len(df), "vlm_unscored": 0},
                     vs.holdout_check(df, df.assign(split="test"))[0], res, None, 0, sens, seeds)
    assert "Verdict under the pre-stated rule" in page and "Sensitivity analyses (not the primary result)" in page
