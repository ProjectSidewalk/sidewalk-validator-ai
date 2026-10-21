"""Tests for analysis/vlm_baseline.py (issue #16). CPU only, synthetic data; the MLX path is importorskip'd."""
import doctest
import os

import numpy as np
import pandas as pd
import pytest

from analysis import vlm_baseline as vb


def test_doctests():
    res = doctest.testmod(vb)
    assert res.failed == 0 and res.attempted > 0


# ---- prompts -------------------------------------------------------------------------------------------------------

def test_build_prompt_has_definition_centre_and_question():
    for t in vb.LABEL_TYPES:
        p = vb.build_prompt(t)
        assert p.startswith(vb.DEFINITIONS[t])
        assert vb.CENTRE_SENTENCE in p
        assert p.endswith(f"Is this a correct {vb.TYPE_NAMES[t]} label? Answer yes or no.")
        assert vb.DEFINITIONS[t] not in vb.build_prompt(t, "no-definition")


def test_definitions_are_short():
    for t, d in vb.DEFINITIONS.items():
        assert d.count(". ") + 1 <= 3, t   # "about three sentences" per the issue


def test_build_prompt_rejects_unknowns():
    with pytest.raises(ValueError):
        vb.build_prompt("Signal")
    with pytest.raises(ValueError):
        vb.build_prompt("CurbRamp", "few-shot")


# ---- yes/no score aggregation --------------------------------------------------------------------------------------

def test_answer_token_ids_keeps_single_tokens_and_dedupes():
    vocab = {"yes": [10], "Yes": [11], " yes": [10], "YES": [12, 13], " Yes": [14], " YES": []}
    ids, used = vb.answer_token_ids(lambda s: vocab.get(s, [1, 2]), vb.YES_VARIANTS)
    assert ids == [10, 11, 14]
    assert used == ["yes", "Yes", " yes", " Yes"]


def test_yes_no_score_sums_variants_and_normalises():
    p = np.array([0.30, 0.10, 0.05, 0.15, 0.40])   # yes, Yes, no, No, other
    s = vb.yes_no_score(np.log(p), yes_ids=[0, 1], no_ids=[2, 3])
    assert s["p_yes"] == pytest.approx(0.40)
    assert s["p_no"] == pytest.approx(0.20)
    assert s["answer_mass"] == pytest.approx(0.60)
    assert s["p_incorrect"] == pytest.approx(0.20 / 0.60)
    assert s["p_correct"] == pytest.approx(0.40 / 0.60)
    assert s["log_odds_incorrect"] == pytest.approx(np.log(0.2 / 0.4))


def test_yes_no_score_is_invariant_to_normalisation_and_resolves_extremes():
    logits = np.array([30.0, -10.0, 5.0])          # unnormalised; yes overwhelmingly likely
    a = vb.yes_no_score(logits, [0], [1])
    b = vb.yes_no_score(logits - 100.0, [0], [1])
    assert a["p_incorrect"] == pytest.approx(b["p_incorrect"])
    assert 0 < a["p_incorrect"] < 1e-15            # still resolved, not rounded to 0 by a ratio of probabilities
    with pytest.raises(ValueError):
        vb.yes_no_score(logits, [], [1])


def test_greedy_agrees_with_score_argmax_on_a_stub():
    """A stub "model": greedy decoding takes the argmax token of the same distribution the score reads. With one
    token per answer, wherever the argmax is a yes/no token the greedy word and the score's verdict must agree."""
    rng = np.random.default_rng(0)
    vocab = ["yes", "no", "The", "maybe"]
    checked = 0
    for _ in range(200):
        logits = rng.normal(size=len(vocab)) * 3
        s = vb.yes_no_score(logits, [0], [1])
        agrees = vb.greedy_agrees(vocab[int(np.argmax(logits))], s["p_incorrect"])
        if np.argmax(logits) in (0, 1):
            assert agrees is True
            checked += 1
        else:
            assert agrees is None
    assert checked > 50


def test_summed_variants_can_disagree_with_greedy():
    """Why the greedy check is run on real crops: summing spellings can flip the verdict against the single top
    token (yes 0.30 is the argmax, but No + no = 0.40 > Yes + yes = 0.35)."""
    p = np.array([0.30, 0.05, 0.20, 0.20, 0.25])    # yes, Yes, no, No, other
    s = vb.yes_no_score(np.log(p), [0, 1], [2, 3])
    assert s["p_incorrect"] > 0.5 and vb.greedy_agrees("yes", s["p_incorrect"]) is False


def test_parse_answer_and_verdict():
    assert vb.parse_answer("**Yes**") == "yes"
    assert vb.parse_answer("") is None
    assert vb.verdict_from_score([0.5, 0.51, 0.2]).tolist() == ["correct", "incorrect", "correct"]


# ---- report helpers ------------------------------------------------------------------------------------------------

def _sample(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    cities = ["seattle-wa", "chicago-il", "secret-a", "secret-b"]
    for t in vb.LABEL_TYPES:
        for i in range(n_per):
            rows.append({"label_uid": f"{cities[i % 4]}:{t}{i}", "city": cities[i % 4], "label_type": t,
                         "label_class": "incorrect" if i % 2 else "correct",
                         "split": "test_heldout" if i % 5 == 0 else "test", "hard": int(i % 7 == 0)})
    s = pd.DataFrame(rows)
    return s, rng


def _scores(sample, rng, model, signal):
    y = (sample.label_class == "incorrect").to_numpy()
    p = np.clip(0.5 + signal * (y - 0.5) + rng.normal(0, 0.25, len(y)), 0.001, 0.999)
    return pd.DataFrame({"label_uid": sample.label_uid, "model": model, "variant": "definition",
                         "p_incorrect": p, "p_correct": 1 - p, "latency_s": rng.uniform(0.5, 2, len(y)),
                         "answer_mass": 0.9})


CITIES = {"seattle-wa": "public", "chicago-il": "public", "secret-a": "private"}   # secret-b: unlisted


def test_predictions_joins_and_rejects_duplicates():
    s, rng = _sample()
    sc = _scores(s, rng, "m", 0.5)
    p = vb.predictions(s, sc)
    assert len(p) == len(s) and (p.true == p.label_class).all()
    with pytest.raises(ValueError):
        vb.predictions(s, pd.concat([sc, sc.iloc[:1]]))


def test_metrics_table_matches_evaluate_and_has_both_classes():
    s, rng = _sample()
    p = vb.predictions(s, _scores(s, rng, "m", 0.6))
    t = vb.metrics_table(p, ["model", "label_type"])
    assert len(t) == 5
    one = p[p.label_type == "CurbRamp"]
    m = vb.evaluate.slice_metrics(one)
    row = t[t.label_type == "CurbRamp"].iloc[0]
    assert row.auc == pytest.approx(m["auc"])
    assert row.balanced_accuracy == pytest.approx(m["balanced_accuracy"])
    for c in ("correct", "incorrect"):
        for lvl in (92, 95):
            assert f"r_{c}_p{lvl}" in t.columns


def test_per_city_names_public_only_and_pools_private():
    s, rng = _sample(n_per=80)
    p = vb.predictions(s, _scores(s, rng, "m", 0.6))
    t = vb.per_city_table(p, CITIES, min_n=30)
    names = set(t.city)
    assert {"seattle-wa", "chicago-il"} <= names
    assert "secret-a" not in names and "secret-b" not in names
    assert "other (2 private deployments)" in names
    assert t[t.city.str.startswith("other")].n.iloc[0] == (p.city.isin(["secret-a", "secret-b"])).sum()
    t2 = vb.per_city_table(p, CITIES, min_n=10_000)
    assert set(t2.city) == {"other (2 private deployments)"}   # small public cities dropped, private row kept


def test_paired_bootstrap_auc_orders_models_and_brackets_point_estimate():
    s, rng = _sample(n_per=100)
    good = _scores(s, rng, "good", 0.8).p_incorrect
    bad = _scores(s, rng, "bad", 0.05).p_incorrect
    y = s.label_class == "incorrect"
    r = vb.paired_bootstrap_auc(y, good, bad, n_boot=200, seed=1)
    assert r["auc_a"] > r["auc_b"]
    assert r["a_lo"] <= r["auc_a"] <= r["a_hi"] and r["diff_lo"] <= r["diff"] <= r["diff_hi"]
    assert r["diff_lo"] > 0
    assert vb.paired_bootstrap_auc(y, good, bad, n_boot=50, seed=3) == vb.paired_bootstrap_auc(y, good, bad,
                                                                                               n_boot=50, seed=3)


def test_best_model_excludes_production_and_latency_table():
    h = pd.DataFrame({"model": ["production", "a", "a", "b", "b"], "auc": [0.99, 0.7, 0.8, 0.6, 0.85],
                      "label_type": ["x"] * 5})
    assert vb.best_model(h) == "a"
    sc = pd.DataFrame({"model": ["a"] * 4, "variant": "definition", "latency_s": [1.0, 2.0, np.nan, 3.0],
                       "answer_mass": [0.9] * 4})
    lat = vb.latency_table(sc).iloc[0]
    assert lat.latency_median_s == 2.0 and lat.n == 4


def test_markdown_table():
    out = vb.markdown(pd.DataFrame({"a": ["x"], "b": [0.12345], "c": [None]}))
    assert out.splitlines()[2] == "| x | 0.123 |  |"


# ---- explanation items and packet blinding -------------------------------------------------------------------------

def _explain_inputs():
    s, rng = _sample(n_per=80)
    best = vb.predictions(s, _scores(s, rng, "best", 0.3))
    prod = vb.predictions(s, _scores(s, rng, "production", 0.3))
    return best, prod


def test_select_items_public_only_balanced_and_categorised():
    best, prod = _explain_inputs()
    items = vb.select_explanation_items(best, prod, CITIES, n_total=20, seed=0)
    assert len(items) == 20
    assert set(items.city) <= {"seattle-wa", "chicago-il"}
    assert items.groupby("label_type").size().eq(4).all()
    assert not items.label_uid.duplicated().any()
    v = vb.verdict_from_score(items.p_incorrect)
    pv = vb.verdict_from_score(items.p_incorrect_prod)
    wrong = items.category == vb.EXPLAIN_CATEGORIES[0]
    assert (v[wrong] != items.true[wrong]).all()
    agree = items.category == vb.EXPLAIN_CATEGORIES[1]
    assert (v[agree] == items.true[agree]).all() and (pv[agree] != items.true[agree]).all()
    again = vb.select_explanation_items(best, prod, CITIES, n_total=20, seed=0)
    assert again.label_uid.tolist() == items.label_uid.tolist()


def test_select_items_falls_back_when_a_pool_is_short():
    best, prod = _explain_inputs()
    prod = prod.assign(p_incorrect=np.where(prod.true == "incorrect", 0.99, 0.01))   # production never wrong
    items = vb.select_explanation_items(best, prod, CITIES, n_total=20, seed=0)
    assert (items.category != vb.EXPLAIN_CATEGORIES[1]).all()
    assert len(items) <= 20 and (items.category == vb.EXPLAIN_CATEGORIES[0]).any()


def _explanations(n=12):
    return pd.DataFrame({"label_uid": [f"seattle-wa:{i}" for i in range(n)],
                         "label_type": [vb.LABEL_TYPES[i % 5] for i in range(n)],
                         "category": [vb.EXPLAIN_CATEGORIES[i % 2] for i in range(n)],
                         "label_class": ["correct", "incorrect"] * (n // 2), "vlm_model": "best",
                         "vlm_answer": ["Yes", "No", "No", "Yes"] * (n // 4),
                         "vlm_reason": [f"reason {i}" for i in range(n)],
                         "p_incorrect": np.linspace(0.1, 0.9, n), "p_incorrect_prod": 0.5})


def test_packet_key_ids_are_random_hex_and_deterministic_in_seed():
    ex = _explanations()
    k1, k2, k3 = vb.packet_key(ex, 7), vb.packet_key(ex, 7), vb.packet_key(ex, 8)
    assert k1.packet_id.str.fullmatch(r"p[0-9a-f]{6}").all() and k1.packet_id.is_unique
    assert k1.packet_id.tolist() == k2.packet_id.tolist()
    assert k1.packet_id.tolist() != k3.packet_id.tolist()
    assert sorted(k1.label_uid) == sorted(ex.label_uid)


def test_blank_sheet_leaks_no_truth_category_score_or_uid():
    ex = _explanations()
    sheet = vb.blank_sheet(vb.packet_key(ex, 7))
    assert list(sheet.columns) == vb.SHEET_COLUMNS
    for forbidden in ("label_uid", "category", "label_class", "p_incorrect", "p_incorrect_prod", "city",
                      "label_id", "vlm_model"):
        assert forbidden not in sheet.columns
    text = sheet.to_csv(index=False)
    assert "seattle-wa" not in text and "vlm_disagrees" not in text and "0.1" not in text
    assert (sheet[["reason_correct", "would_change_my_vote", "my_verdict"]] == "").all().all()


def test_unblind_and_agreement_round_trip():
    ex = _explanations(8)
    key = vb.packet_key(ex, 3)
    sheet = vb.blank_sheet(key)
    # the rater agrees with the crowd everywhere
    truth = dict(zip(key.packet_id, key.label_class))
    sheet["my_verdict"] = sheet.packet_id.map(truth)
    sheet["reason_correct"] = "yes"
    sheet["would_change_my_vote"] = "No "
    vb.validate_codes(sheet)
    t = vb.agreement_tables(vb.unblind(sheet, key))
    overall = t[t.category == "all"].iloc[0]
    assert overall.n == 8 and overall.rater_agrees_crowd == 1.0 and overall.reason_correct_yes == 1.0
    vlm_correct = np.where(key.vlm_answer.str.lower() == "no", "incorrect", "correct")
    assert overall.rater_agrees_vlm == pytest.approx((vlm_correct == key.label_class).mean())
    bad = sheet.copy()
    bad.loc[0, "my_verdict"] = "probably"
    with pytest.raises(ValueError):
        vb.validate_codes(bad)
    extra = pd.concat([sheet, sheet.iloc[:1].assign(packet_id="p000000")])
    with pytest.raises(ValueError):
        vb.unblind(extra, key)


def test_key_must_be_outside_the_repo(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(vb.__file__)))
    assert vb.inside_this_repo(os.path.join(here, "docs", "packet_key.csv"))
    assert not vb.inside_this_repo(str(tmp_path / "key.csv"))


def test_render_item(tmp_path):
    from PIL import Image

    src = tmp_path / "c.webp"
    Image.new("RGB", (640, 640), (90, 90, 90)).save(src)
    out = tmp_path / "item.jpg"
    vb.render_item(str(src), "001  p123abc  label type: curb ramp", "No", "The ramp is a driveway " * 4, str(out))
    im = Image.open(out)
    assert im.width == 640 and im.height > 640


def test_mlx_scorer_imports():
    pytest.importorskip("mlx_vlm")
    assert callable(vb.MlxVlm)
