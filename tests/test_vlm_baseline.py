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


def test_best_model_excludes_production():
    h = pd.DataFrame({"model": ["production", "a", "a", "b", "b"], "auc": [0.99, 0.7, 0.8, 0.6, 0.85],
                      "label_type": ["x"] * 5})
    assert vb.best_model(h) == "a"


def test_timing_table_reads_runs_and_load(tmp_path):
    pd.DataFrame({"label_uid": list("abcde"), "label_type": "CurbRamp", "latency_s": [1.0, 2.0, 3.0, 4.0, 5.0],
                  "reason_latency_s": 2.0}).to_csv(tmp_path / "m.csv", index=False)
    (tmp_path / "m.meta.json").write_text('{"load_before": [1.5, 2, 3], "load_after": [1.7, 2, 3], "date": "d"}')
    t = vb.timing_table(str(tmp_path)).iloc[0]
    assert t.model == "m" and t.n == 5 and t.median_s == 3.0 and t.p10_s == pytest.approx(1.4)
    assert t.reason_median_s == 2.0 and t.load_1min_before == 1.5
    assert vb.timing_table(str(tmp_path / "missing")).empty


def test_stratified_crops_cover_every_type_and_class():
    s, _ = _sample(n_per=80)
    s["split"] = np.where(np.arange(len(s)) % 2, "test", "test_heldout")
    s["label_class"] = np.where((np.arange(len(s)) // 2) % 2, "incorrect", "correct")
    c = vb.stratified_crops(s, 2, subset=20)
    assert c.groupby(["label_type", "label_class"]).size().eq(1).all() and len(c) == 10


def test_within_city_auc_pair_weighted_and_ignores_between_city_shift():
    # city a: balanced, perfect (AUC 1); city b: 1 incorrect of 20, inverted (AUC 0) -> weights 100 vs 19
    y = [0, 1] * 10 + [1] + [0] * 19
    score = [0, 1] * 10 + [0] + [1] * 19
    city = ["a"] * 20 + ["b"] * 20
    auc, n = vb.within_city_auc(y, score, city)
    assert n == 2 and auc == pytest.approx(100 / 119)
    # a score that only separates a high-incorrect city from a low-incorrect one has no within-city skill
    y2 = [1] * 15 + [0] * 5 + [1] * 5 + [0] * 15
    s2 = [1.0] * 20 + [0.0] * 20
    assert vb.within_city_auc(y2, s2, city)[0] == pytest.approx(0.5)


def test_cluster_bootstrap_is_wider_than_label_bootstrap_when_cities_differ():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(8):
        skill = 0.9 if c % 2 else 0.1        # the model's advantage flips between cities
        for i in range(60):
            y = i % 2
            rows.append((f"c{c}", y, y * skill + rng.normal(0, 0.3), rng.normal(0, 0.3) + 0.3 * y))
    d = pd.DataFrame(rows, columns=["city", "y", "a", "b"])
    lab = vb.paired_bootstrap_auc(d.y, d.a, d.b, n_boot=200, seed=1)
    clu = vb.cluster_bootstrap_auc(d.y, d.a, d.b, d.city, n_boot=200, seed=1)
    assert clu["n_groups"] == 8
    assert clu["diff_hi"] - clu["diff_lo"] > lab["diff_hi"] - lab["diff_lo"]


def test_split_diff_bootstrap_brackets_the_difference():
    s, rng = _sample(n_per=100)
    p = vb.predictions(s, _scores(s, rng, "m", 0.6))
    r = vb.split_diff_bootstrap(p, n_boot=200, seed=0)
    assert r["diff"] == pytest.approx(r["auc_test"] - r["auc_heldout"])
    assert r["diff_lo"] <= r["diff"] <= r["diff_hi"] and r["n_test"] + r["n_heldout"] == len(p)


def test_per_city_type_and_mean_within_type():
    s, rng = _sample(n_per=80)
    p = vb.predictions(s, _scores(s, rng, "m", 0.6))
    t = vb.per_city_type_table(p, CITIES, min_n=20)
    assert set(t.city) == {"seattle-wa", "chicago-il", "other (2 private deployments)"}
    assert t.groupby("label_type").size().eq(3).all()
    assert vb.per_city_type_table(p, CITIES, min_n=10_000).city.str.startswith("other").all()
    c = vb.per_city_table(p, CITIES, min_n=30)
    # the fixture gives each public city a single class, so only the pooled private row has within-type AUCs
    assert {"mean_within_type_auc", "n_types"} <= set(c.columns)
    assert c.set_index("city").n_types.to_dict() == {"seattle-wa": 0, "chicago-il": 0,
                                                     "other (2 private deployments)": 5}


def test_check_coverage_rejects_partial_runs():
    s, rng = _sample()
    p = vb.predictions(s, pd.concat([_scores(s, rng, "full", 0.5), _scores(s.iloc[:30], rng, "part", 0.5)]))
    vb.check_coverage(p, set(s.label_uid[:20]))                    # both cover these 20
    with pytest.raises(ValueError, match="part"):
        vb.check_coverage(p, set(s.label_uid[:50]))


def test_markdown_table():
    out = vb.markdown(pd.DataFrame({"a": ["x"], "b": [0.12345], "c": [None]}))
    assert out.splitlines()[2] == "| x | 0.123 |  |"


# ---- explanation items and packet blinding -------------------------------------------------------------------------

def _explain_inputs():
    s, rng = _sample(n_per=80)
    best = vb.predictions(s, _scores(s, rng, "best", 0.3))
    prod = vb.predictions(s, _scores(s, rng, "production", 0.3))
    return best, prod


def test_select_items_answer_and_correctness_say_nothing_about_the_crowd():
    best, prod = _explain_inputs()
    items = vb.select_explanation_items(best, prod, CITIES, per_type=8, seed=0)
    assert set(items.city) <= {"seattle-wa", "chicago-il"}
    assert not items.label_uid.duplicated().any()
    assert items.groupby("label_type").size().le(8).all() and len(items) > 20
    v = vb.verdict_from_score(items.p_incorrect)
    pv = vb.verdict_from_score(items.p_incorrect_prod)
    assert ((v == items.true) == items.vlm_correct).all()
    assert (items.vlm_answer == np.where(v == "incorrect", "No", "Yes")).all()
    assert (items.category == vb.item_category(v != items.true, pv != items.true)).all()
    assert items.groupby("label_type").vlm_correct.mean().eq(0.5).all()
    crowd_correct = (items.true == "correct")
    assert crowd_correct.groupby([items.label_type, items.vlm_answer]).mean().eq(0.5).all()
    # within each cell, production-wrong rows are taken before production-right ones
    pool = best.merge(prod[["label_uid", "p_incorrect"]], on="label_uid", suffixes=("", "_p"))
    pool = pool[pool.city.isin(["seattle-wa", "chicago-il"])]
    pv_all = vb.verdict_from_score(pool.p_incorrect_p) != pool.true
    pool = pool.assign(prod_wrong=pv_all, answer=np.where(vb.verdict_from_score(pool.p_incorrect) == "incorrect",
                                                          "No", "Yes"),
                       right=vb.verdict_from_score(pool.p_incorrect) == pool.true)
    chosen = set(items.label_uid)
    for _, g in pool.groupby(["label_type", "answer", "right"]):
        sel, rest = g[g.label_uid.isin(chosen)], g[~g.label_uid.isin(chosen)]
        if len(sel) and rest.prod_wrong.any():
            assert sel.prod_wrong.all()
    again = vb.select_explanation_items(best, prod, CITIES, per_type=8, seed=0)
    assert again.label_uid.tolist() == items.label_uid.tolist()


def test_select_items_prefers_kept_rows():
    best, prod = _explain_inputs()
    first = vb.select_explanation_items(best, prod, CITIES, per_type=8, seed=0)
    keep = first.label_uid.iloc[::3].tolist()
    second = vb.select_explanation_items(best, prod, CITIES, per_type=8, seed=99, keep=keep)
    assert set(keep) <= set(second.label_uid)


def _explanations(n=12):
    return pd.DataFrame({"label_uid": [f"seattle-wa:{i}" for i in range(n)],
                         "label_type": [vb.LABEL_TYPES[i % 5] for i in range(n)],
                         "category": [vb.CATEGORIES[i % 4] for i in range(n)],
                         "vlm_correct": [i % 4 >= 2 for i in range(n)],
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


def test_sheets_leak_no_truth_category_score_or_uid_and_pass1_hides_the_answer():
    ex = _explanations()
    pass1, pass2 = vb.blank_sheets(vb.packet_key(ex, 7))
    assert list(pass1.columns) == vb.PASS1_COLUMNS and list(pass2.columns) == vb.PASS2_COLUMNS
    for sheet in (pass1, pass2):
        for forbidden in ("label_uid", "category", "vlm_correct", "label_class", "p_incorrect", "p_incorrect_prod",
                          "city", "label_id", "vlm_model"):
            assert forbidden not in sheet.columns
        text = sheet.to_csv(index=False)
        assert "seattle-wa" not in text and "vlm_wrong" not in text and "0.1" not in text
    assert "vlm_answer" not in pass1 and "vlm_reason" not in pass1 and "reason" not in pass1.to_csv(index=False)
    assert (pass1.my_verdict == "").all() and (pass2[["reason_correct", "would_change_my_vote"]] == "").all().all()


def test_rubrics_say_nothing_about_composition_and_pass1_says_nothing_about_the_model():
    for text in (vb.RUBRIC_PASS1_TEXT.lower(), vb.RUBRIC_PASS2_TEXT.lower()):
        for word in ("disagree", "crowd", "production", "category", "half", "balanced", "wrong"):
            assert word not in text, word
    p1 = vb.RUBRIC_PASS1_TEXT.lower()
    for word in ("model", "answer", "reason", "vlm", "items_annotated", "pass2_sheet"):
        assert word not in p1, word
    assert "pass 1" in p1 and "pass 2" in vb.RUBRIC_PASS2_TEXT.lower()


def test_packet_writes_two_zips_with_pass2_material_only_in_the_second(tmp_path):
    import zipfile

    from PIL import Image

    sample_dir, out_dir = tmp_path / "sample", tmp_path / "out"
    ex = _explanations(8)
    rows = []
    for uid, t in zip(ex.label_uid, ex.label_type):
        crop = f"correct/{uid.replace(':', '_')}.webp"
        (sample_dir / "crops" / t / "correct").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (90, 90, 90)).save(sample_dir / "crops" / t / crop)
        rows.append({"label_uid": uid, "label_id": uid.split(":")[1], "label_type": t, "crop_path": crop})
    pd.DataFrame(rows).to_csv(sample_dir / "sample.csv", index=False)
    out_dir.mkdir()
    ex.assign(reason_source="generated", n_words=3, latency_s=1.0).to_csv(out_dir / "explanations.csv", index=False)
    key = tmp_path / "private" / "packet_key.csv"
    args = ["packet", "--sample-dir", str(sample_dir), "--out-dir", str(out_dir), "--packet-dir",
            str(tmp_path / "packet"), "--key", str(key), "--seed", "5", "--doc-dir", str(tmp_path / "doc")]
    vb.main(args)
    z1 = zipfile.ZipFile(out_dir / vb.PACKET_ZIPS[1]).namelist()
    z2 = zipfile.ZipFile(out_dir / vb.PACKET_ZIPS[2]).namelist()
    assert {n for n in z1 if not n.endswith(".jpg") and not n.endswith("/")} == {"pass1_sheet.csv", "RUBRIC.md",
                                                                                 "rate.html"}
    assert {n for n in z2 if not n.endswith(".jpg") and not n.endswith("/")} == {"pass2_sheet.csv", "RUBRIC.md",
                                                                                 "rate.html"}
    jpg1 = sorted(n.split("/")[-1] for n in z1 if n.endswith(".jpg"))
    jpg2 = sorted(n.split("/")[-1] for n in z2 if n.endswith(".jpg"))
    assert jpg1 == jpg2 and len(jpg1) == 8                      # same items and ids in both
    assert all(n.startswith("items_bare/") for n in z1 if n.endswith(".jpg"))
    assert all(n.startswith("items_annotated/") for n in z2 if n.endswith(".jpg"))
    with zipfile.ZipFile(out_dir / vb.PACKET_ZIPS[1]) as z:
        sheet1 = z.read("pass1_sheet.csv").decode()
        assert "vlm_answer" not in sheet1 and "reason" not in sheet1
        assert z.read("RUBRIC.md").decode() == vb.RUBRIC_PASS1_TEXT
    digest = vb.sha256_of(str(key))
    vb.main(args)                                               # same seed and items: the key is unchanged
    assert vb.sha256_of(str(key)) == digest
    assert (tmp_path / "doc" / "packet_key.sha256").read_text().split()[0] == digest


def _built_packet(tmp_path):
    """A packet built by ``packet`` on 8 synthetic items; returns (args used, key path, explanations)."""
    from PIL import Image

    sample_dir, out_dir = tmp_path / "sample", tmp_path / "out"
    ex = _explanations(8)
    rows = []
    for uid, t in zip(ex.label_uid, ex.label_type):
        crop = f"correct/{uid.replace(':', '_')}.webp"
        (sample_dir / "crops" / t / "correct").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (90, 90, 90)).save(sample_dir / "crops" / t / crop)
        rows.append({"label_uid": uid, "label_id": uid.split(":")[1], "label_type": t, "crop_path": crop})
    pd.DataFrame(rows).to_csv(sample_dir / "sample.csv", index=False)
    out_dir.mkdir()
    ex.assign(reason_source="generated", n_words=3, latency_s=1.0).to_csv(out_dir / "explanations.csv", index=False)
    key = tmp_path / "private" / "packet_key.csv"
    vb.main(["packet", "--sample-dir", str(sample_dir), "--out-dir", str(out_dir), "--packet-dir",
             str(tmp_path / "packet"), "--key", str(key), "--seed", "5", "--doc-dir", str(tmp_path / "doc")])
    return out_dir, key, ex


def _meta(page):
    import json
    import re

    return json.loads(re.search(r'<script id="meta" type="application/json">(.*?)</script>', page, re.S)
                      .group(1).replace("<\\/", "</"))


def test_rating_pages_keep_pass1_blind_and_match_the_sheets(tmp_path):
    import zipfile

    out_dir, key, ex = _built_packet(tmp_path)
    p1 = (tmp_path / "packet" / "pass1" / "rate.html").read_text()
    p2 = (tmp_path / "packet" / "pass2" / "rate.html").read_text()
    for reason in ex.vlm_reason:
        assert reason not in p1
    assert "vlm_answer" not in p1 and "items_annotated" not in p1 and "Model" not in p1
    assert 'id="gatefile"' not in p1 and 'id="gatefile"' in p2
    sheet1 = pd.read_csv(tmp_path / "packet" / "pass1" / "pass1_sheet.csv", dtype=str, keep_default_na=False)
    sheet2 = pd.read_csv(tmp_path / "packet" / "pass2" / "pass2_sheet.csv", dtype=str, keep_default_na=False)
    m1, m2 = _meta(p1), _meta(p2)
    assert m1["columns"] == vb.PASS1_COLUMNS and m2["columns"] == vb.PASS2_COLUMNS
    assert [r["packet_id"] for r in m1["rows"]] == list(sheet1.packet_id) == list(sheet2.packet_id)
    assert [r["vlm_reason"] for r in m2["rows"]] == list(sheet2.vlm_reason)
    assert m2["gate"]["column"] == "my_verdict" and m1["gate"] is None
    for i, pid in enumerate(sheet1.packet_id):                  # every card points at a crop that exists
        assert f'src="items_bare/{i + 1:03d}_{pid}.jpg"' in p1
        assert (tmp_path / "packet" / "pass1" / "items_bare" / f"{i + 1:03d}_{pid}.jpg").exists()
    assert "rate.html" in zipfile.ZipFile(out_dir / vb.PACKET_ZIPS[1]).namelist()
    assert "rate.html" in zipfile.ZipFile(out_dir / vb.PACKET_ZIPS[2]).namelist()


def test_gallery_rebuilds_pages_on_an_old_packet_and_refuses_a_wrong_key(tmp_path):
    out_dir, key, _ = _built_packet(tmp_path)
    for n in (1, 2):
        (tmp_path / "packet" / f"pass{n}" / "rate.html").unlink()
    digest = vb.sha256_of(str(key))
    args = ["gallery", "--out-dir", str(out_dir), "--packet-dir", str(tmp_path / "packet"), "--key", str(key),
            "--key-sha256", str(tmp_path / "doc" / "packet_key.sha256"), "--doc-dir", str(tmp_path / "doc")]
    vb.main(args)
    assert (tmp_path / "packet" / "pass2" / "rate.html").exists()
    assert vb.sha256_of(str(key)) == digest
    import json
    prov = json.load(open(tmp_path / "doc" / "provenance.json"))
    assert "rate.html" in prov["packet"]["zips"][vb.PACKET_ZIPS[1]]["files"]
    other = tmp_path / "other.sha256"
    other.write_text("0" * 64 + "  packet_key.csv\n")
    with pytest.raises(SystemExit):
        vb.main(args[:-4] + ["--key-sha256", str(other), "--doc-dir", ""])
    sheet = tmp_path / "packet" / "pass1" / "pass1_sheet.csv"
    rows = sheet.read_text().splitlines()
    sheet.write_text("\n".join([rows[0], rows[2], rows[1]] + rows[3:]) + "\n")   # swap two items
    with pytest.raises(SystemExit):
        vb.main(args)


def test_unblind_and_agreement_round_trip():
    ex = _explanations(8)
    key = vb.packet_key(ex, 3)
    pass1, pass2 = vb.blank_sheets(key)
    truth = dict(zip(key.packet_id, key.label_class))      # the rater agrees with the crowd everywhere
    pass1["my_verdict"] = pass1.packet_id.map(truth)
    pass2["reason_correct"] = "yes"
    pass2["would_change_my_vote"] = "No "
    vb.validate_codes(pass1)
    vb.validate_codes(pass2)
    t = vb.agreement_tables(vb.unblind(pass1, pass2, key))
    overall = t[t.by == "all"].iloc[0]
    assert overall.n == 8 and overall.rater_agrees_crowd == 1.0 and overall.reason_correct_yes == 1.0
    vlm_verdict = np.where(key.vlm_answer.str.lower() == "no", "incorrect", "correct")
    assert overall.rater_agrees_vlm == pytest.approx((vlm_verdict == key.label_class).mean())
    assert set(t[t.by == "category"].group) == set(key.category)
    assert set(t[t.by == "vlm"].group) <= {"vlm right", "vlm wrong"}
    bad = pass1.copy()
    bad.loc[0, "my_verdict"] = "probably"
    with pytest.raises(ValueError):
        vb.validate_codes(bad)
    with pytest.raises(ValueError):
        vb.unblind(pass1.iloc[1:], pass2, key)
    extra = pd.concat([pass1, pass1.iloc[:1].assign(packet_id="p000000")])
    extra2 = pd.concat([pass2, pass2.iloc[:1].assign(packet_id="p000000")])
    with pytest.raises(ValueError):
        vb.unblind(extra, extra2, key)


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
    bare = tmp_path / "bare.jpg"
    vb.render_item(str(src), "001  p123abc  label type: curb ramp", None, None, str(bare))
    im, b = Image.open(out), Image.open(bare)
    assert im.width == 640 and im.height > b.height > 640


def test_mlx_scorer_imports():
    pytest.importorskip("mlx_vlm")
    assert callable(vb.MlxVlm)
