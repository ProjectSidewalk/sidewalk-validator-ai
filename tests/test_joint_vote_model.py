"""Tests for analysis/joint_vote_model.py: the EM on synthetic data where the truth is known, the expert-vote
exclusion, the nested cross-validation, the paired statistics, the rule, and the binding of the committed doc to its
JSON. Synthetic data only; no network, no ``~/Data``."""
import doctest
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import joint_vote_model as jv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "joint-vote-model-2026-11.md")
JSON = os.path.join(ROOT, "docs", "joint-vote-model-2026-11", "results.json")


def test_doctests():
    assert doctest.testmod(jv).failed == 0


def _fit(d, kappa, **kw):
    return jv.fit_ds(d["label"], d["validator"], d["agree"], d["n_labels"], d["n_validators"], d["group"],
                     int(d["group"].max()) + 1, kappa, **kw)


def _majority_acc(d):
    a = np.bincount(d["label"], d["agree"], d["n_labels"])
    n = np.bincount(d["label"], minlength=d["n_labels"])
    decided = 2 * a != n
    return ((2 * a > n) == d["truth"])[decided].mean(), decided


# ---------------------------------------------------------------------------------------------------------------------
# The EM against a known truth
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_em_beats_majority_on_the_labels_majority_decides(seed):
    d = jv.synthetic(n_labels=4000, n_validators=80, votes_per_label=(2, 5), seed=seed)
    f = _fit(d, kappa=5)
    acc_maj, decided = _majority_acc(d)
    acc_ds = ((f["q"] > 0.5) == d["truth"])[decided].mean()
    assert f["converged"]
    assert acc_ds > acc_maj + 0.003  # measured +0.4 to +1.8 points over these seeds


def test_two_lax_agrees_lose_to_one_careful_disagree():
    """The asymmetric case the model exists for: a 2-1 Agree majority made of validators who agree with everything,
    against one careful Disagree, is called incorrect (the likelihood ratio is about 1.29^2 x 0.12 = 0.2)."""
    d = jv.synthetic(n_labels=20000, n_validators=40, votes_per_label=(3, 6), seed=8)
    base_n = d["n_labels"]
    lax = np.flatnonzero(np.arange(40) % 3 == 0)
    careful = np.flatnonzero(np.arange(40) % 3 != 0)
    lab = np.r_[d["label"], base_n, base_n, base_n]
    val = np.r_[d["validator"], lax[0], lax[1], careful[0]]
    agree = np.r_[d["agree"], 1, 1, 0]
    f = jv.fit_ds(lab, val, agree, base_n + 1, 40, np.zeros(base_n + 1, int), 1, 5)
    assert f["q"][-1] < 0.5
    # three careful validators 2-1 the same way stay correct
    val2 = np.r_[d["validator"], careful[1], careful[2], careful[0]]
    g = jv.fit_ds(lab, val2, agree, base_n + 1, 40, np.zeros(base_n + 1, int), 1, 5)
    assert g["q"][-1] > 0.5


def test_em_recovers_the_validator_rates():
    d = jv.synthetic(n_labels=20000, n_validators=40, votes_per_label=(3, 6), seed=3)
    f = _fit(d, kappa=2)
    busy = f["n_votes"] >= 500
    assert busy.sum() >= 10
    assert np.abs(f["s"][busy] - d["s"][busy]).max() < 0.06
    assert np.abs(f["p"][busy] - d["p"][busy]).max() < 0.08
    # the lax validators are found as lax: high sensitivity, low specificity
    lax = (np.arange(40) % 3 == 0) & busy
    assert (f["p"][lax] < 0.45).all() and (f["p"][busy & ~lax] > 0.7).all()


def test_class_prior_is_recovered_per_group():
    rng = np.random.default_rng(4)
    d = jv.synthetic(n_labels=6000, n_validators=60, votes_per_label=(3, 5), seed=4)
    # make group 1 mostly incorrect by flipping its truth and regenerating the votes
    group = (np.arange(d["n_labels"]) >= 3000).astype(int)
    truth = np.where(group == 1, rng.random(d["n_labels"]) < 0.3, d["truth"])
    pa = np.where(truth[d["label"]], d["s"][d["validator"]], 1 - d["p"][d["validator"]])
    agree = (rng.random(len(pa)) < pa).astype(int)
    f = jv.fit_ds(d["label"], d["validator"], agree, d["n_labels"], d["n_validators"], group, 2, 5)
    assert abs(f["pi"][0] - truth[group == 0].mean()) < 0.05
    assert abs(f["pi"][1] - truth[group == 1].mean()) < 0.05


def test_large_kappa_shrinks_every_validator_to_the_population():
    d = jv.synthetic(n_labels=3000, n_validators=50, seed=5)
    f = _fit(d, kappa=1e7)
    assert np.ptp(f["s"]) < 1e-3 and np.ptp(f["p"]) < 1e-3
    assert abs(f["s"][0] - f["mu_s"]) < 1e-3


def test_unvoted_label_keeps_its_prior_and_labeler_arm_runs():
    d = jv.synthetic(n_labels=500, n_validators=20, seed=6)
    n = d["n_labels"] + 1  # one extra label with no vote
    group = np.zeros(n, int)
    f = jv.fit_ds(d["label"], d["validator"], d["agree"], n, d["n_validators"], group, 1, 5)
    assert f["q"][-1] == pytest.approx(f["prior"][-1])
    labeler = np.r_[np.arange(d["n_labels"]) % 7, -1]
    g = jv.fit_ds(d["label"], d["validator"], d["agree"], n, d["n_validators"], group, 1, 5, labeler=labeler,
                  n_labelers=7)
    assert g["converged"] and np.isfinite(g["q"]).all()


def test_deterministic():
    d = jv.synthetic(n_labels=800, n_validators=30, seed=7)
    assert np.array_equal(_fit(d, 3)["q"], _fit(d, 3)["q"])


# ---------------------------------------------------------------------------------------------------------------------
# Votes and leakage
# ---------------------------------------------------------------------------------------------------------------------

def _validations(*votes):
    return json.dumps([{"user_id": u, "validation": v, "validator_type": t} for u, v, t in votes])


def test_read_snapshot_drops_ai_expert_unsure_votes(tmp_path):
    rows = [
        {"label_id": "1", "user_id": "lab1", "label_type": "CurbRamp",
         "validations": _validations(("x", "Disagree", "AI"), ("e1", "Agree", "Human"), ("u1", "Agree", "Human"),
                                     ("u2", "Unsure", "Human"), ("u3", "Disagree", "Human"))},
        {"label_id": "2", "user_id": "e2", "label_type": "CurbRamp",
         "validations": _validations(("e2", "Disagree", "Human"), ("u1", "Disagree", "Human"))},
        {"label_id": "3", "user_id": "lab1", "label_type": "Signal", "validations": "[]"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "city-a.csv", index=False)
    labels, votes, counts = jv.read_snapshot(str(tmp_path), ["city-a"], ("CurbRamp",), {"e1", "e2"})
    assert labels.label_uid.tolist() == ["city-a:1", "city-a:2"]
    assert labels.a.tolist() == [1, 0] and labels.d.tolist() == [1, 1]
    assert labels.labeler.tolist() == ["lab1", ""]  # an expert labeler gets no labeler parameter
    assert not votes.user_id.isin(["e1", "e2", "x"]).any()
    assert counts == {"ai": 1, "expert": 2, "unsure": 1, "duplicate": 0, "human_ad": 3}


def test_read_snapshot_refuses_duplicate_label_uid(tmp_path):
    rows = [{"label_id": "1", "user_id": "a", "label_type": "CurbRamp", "validations": "[]"}] * 2
    pd.DataFrame(rows).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="not unique"):
        jv.read_snapshot(str(tmp_path), ["c"], ("CurbRamp",), set())


def test_score_refuses_without_experts(monkeypatch, tmp_path):
    monkeypatch.delenv("EXPERTS", raising=False)
    with pytest.raises(SystemExit, match="EXPERTS"):
        jv.main(["score", "--out-dir", str(tmp_path / "out"), "--gold", "missing.csv"])


def test_snapshot_written_after_rating_is_refused(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("x\n")
    with pytest.raises(ValueError, match="first rating day"):
        jv.snapshot_files_ok(str(tmp_path), ["c"], rated_after="2000-01-01")
    jv.snapshot_files_ok(str(tmp_path), ["c"], rated_after="2999-01-01")


# ---------------------------------------------------------------------------------------------------------------------
# Selection and statistics
# ---------------------------------------------------------------------------------------------------------------------

def test_nested_cv_never_scores_a_fold_with_its_own_choice():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    fold = np.arange(100) % 5
    # kappa 1 is perfect on fold 0 only and useless elsewhere; kappa 10 is mediocre everywhere
    q1 = np.where(fold == 0, y, 1 - y).astype(float)
    q10 = np.where(y == 1, 0.6, 0.4)
    q, chosen = jv.nested_cv({1.0: q1, 10.0: q10}, y, fold, np.ones(100, bool), (1.0, 10.0))
    assert chosen[0] == 10.0  # fold 0's own items, where kappa 1 shines, are not used to choose for fold 0
    assert np.allclose(q[fold == 0], q10[fold == 0])


def test_paired_boot_cluster_interval_is_wider_for_clustered_differences():
    d_a = np.repeat([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 10)
    d_b = np.zeros(100)
    groups = np.repeat(np.arange(10), 10)
    _, lo, hi = jv.paired_boot(d_a, d_b, None, 2000, 0)
    _, clo, chi = jv.paired_boot(d_a, d_b, groups, 2000, 0)
    assert (chi - clo) > (hi - lo)


@pytest.mark.parametrize("clustered", [False, True])
def test_nested_boot_with_one_kappa_is_the_plain_paired_bootstrap(clustered):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    q = np.clip(y + rng.normal(0, 0.45, 60), 0.01, 0.99)
    ref = rng.random(60) < 0.6
    groups = np.repeat(np.arange(20), 3) if clustered else None
    iv = jv.nested_boot({5.0: q}, y, np.arange(60) % 5, np.ones(60, bool), ref, {"all": np.ones(60, bool)},
                        groups, (5.0,), 500, 3)
    _, lo, hi = jv.paired_boot((q > 0.5) == (y == 1), ref, groups, 500, 3)
    assert iv["all"] == pytest.approx((lo, hi))


def test_nested_boot_widens_when_the_choice_is_unstable():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    fold = np.arange(200) % 5
    good = np.where(y == 1, 0.7, 0.3)
    noisy = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.5, 200), 0.01, 0.99)
    ref = np.zeros(200)
    one = jv.nested_boot({1.0: good}, y, fold, np.ones(200, bool), ref, {"a": np.ones(200, bool)}, None, (1.0,), 500)
    two = jv.nested_boot({1.0: good, 2.0: noisy}, y, fold, np.ones(200, bool), ref, {"a": np.ones(200, bool)}, None,
                         (1.0, 2.0), 500)
    assert two["a"][1] - two["a"][0] >= one["a"][1] - one["a"][0] - 1e-12


def test_score_end_to_end_on_synthetic_gold():
    d = jv.synthetic(n_labels=1500, n_validators=40, votes_per_label=(2, 5), seed=9)
    uids = [f"city-a:{i}" for i in range(d["n_labels"])]
    a = np.bincount(d["label"], d["agree"], d["n_labels"]).astype(int)
    n = np.bincount(d["label"], minlength=d["n_labels"])
    labels = pd.DataFrame({"label_uid": uids, "city": "city-a", "label_type": "CurbRamp",
                           "labeler": [f"l{i % 9}" for i in range(d["n_labels"])], "a": a, "d": n - a})
    votes = pd.DataFrame({"label_uid": np.array(uids)[d["label"]], "user_id": [f"u{v}" for v in d["validator"]],
                          "agree": d["agree"]})
    rng = np.random.default_rng(0)
    pick = rng.choice(d["n_labels"], 300, replace=False)
    gold = pd.DataFrame({"label_uid": np.array(uids)[pick], "decided": True, "expert_source": "r1",
                         "expert": np.where(d["truth"][pick], "correct", "incorrect"),
                         "ai_vote": np.where(np.arange(300) % 4 == 0, "Agree", "Disagree"),
                         "pano_id": [f"p{i // 2}" for i in range(300)], "round": np.where(np.arange(300) < 150, "1", "2"),
                         "label_type": "CurbRamp", "severity": "2", "tags": "[]", "stratum": "x"})
    pool = pd.DataFrame({"schema": "sidewalk_a", "label_id": [str(i) for i in range(d["n_labels"])],
                         "label_type": "CurbRamp", "ai_vote": "Disagree"})
    j, g = jv.score(gold, labels, votes, pool, {"sidewalk_a": "city-a"}, None, (1.0, 5.0, 50.0), n_boot=200)
    p = j["arms"]["votes"]["primary"]
    assert p["n"] == j["counts"]["primary"] > 0
    assert p["lo"] <= p["diff"] <= p["hi"] and p["cluster_lo"] <= p["diff"] <= p["cluster_hi"]
    assert set(j["arms"]["votes"]["chosen_kappa"]) == {str(k) for k in range(jv.N_FOLDS)}
    assert j["arms"]["votes"]["coverage"]["n_in_snapshot"] == d["n_labels"]
    text = jv.ts.dumps(j)
    assert "u1" not in text and "city-a:" not in text
    jv.render_blocks({**j, "snapshot": {"ai": 0, "expert": 0, "unsure": 0, "duplicate": 0, "human_ad": len(votes),
                                        "n_deployments": 1, "n_labels": len(labels), "n_labels_voted": len(labels)},
                      "provenance": {k: "x" for k in ("gold_sha256", "snapshot_sha256", "pool_sha256",
                                                       "verdicts_sha256", "posteriors_sha256", "git_commit", "python",
                                                       "pandas", "numpy", "cities_read_at")}
                      | {"gold_matches_pin": False, "n_snapshot_files": 1, "n_experts": 2, "n_public": 1}})


def test_rule():
    good = {"diff": 0.05, "lo": 0.01, "hi": 0.09, "cluster_lo": 0.005, "cluster_hi": 0.1}
    assert jv.rule_met(good, {"diff": 0.0})["met"]
    assert not jv.rule_met({**good, "diff": 0.039}, {"diff": 0.0})["met"]
    assert not jv.rule_met({**good, "cluster_lo": -0.001}, {"diff": 0.0})["met"]
    assert not jv.rule_met(good, {"diff": -0.003})["met"]


def test_gold_frame_populations():
    gold = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3", "c:4", "c:5"], "decided": [True] * 4 + [False],
                         "expert_source": ["r1", "r1", "r2", "r2_consensus", "r1"],
                         "expert": ["correct", "incorrect", "correct", "correct", ""],
                         "ai_vote": ["Disagree", "Disagree", "Agree", "Disagree", "Disagree"],
                         "pano_id": list("abcde"), "round": ["1", "1", "2", "2", "1"], "label_type": "CurbRamp"})
    labels = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3", "c:4"], "a": [2, 0, 3, 0], "d": [1, 1, 0, 2]})
    g = jv.gold_frame(gold, labels)
    assert g.label_uid.tolist() == ["c:1", "c:2", "c:3"]  # undecided and meeting verdicts left out
    assert g.primary.tolist() == [True, False, False]     # c:2 has one vote; c:3 is a decoy
    assert g.decisive.tolist() == [False, False, True]


# ---------------------------------------------------------------------------------------------------------------------
# The committed doc
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(JSON), reason="results.json is written by the run")
def test_doc_matches_results_json():
    j = json.load(open(JSON))
    doc = open(DOC).read()
    assert jv.splice(doc, jv.render_blocks(j)) == doc
    text = open(JSON).read()
    for key in ('"label_uid"', '"user_id"', '"raters"', '"label_id"'):
        assert key not in text
