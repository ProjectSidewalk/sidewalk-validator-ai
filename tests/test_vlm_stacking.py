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
    x = np.column_stack([vs.logit(g.p_r3, 1e-6), vs.logit(g.p_vlm, 0.01)])
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
    rows, trained = vs.holdout_check(df, run)
    assert trained.sum() == 2 and rows.set_index("check").n["round-3 split = train"] == 1
    assert "Verdict: YES" in vs.render({"n_both": len(df), "n_vlm": len(df), "n_classifier": len(df),
                                        "vlm_unscored": 0}, rows, res, None, 0)
