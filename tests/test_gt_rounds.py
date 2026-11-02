"""Tests for analysis/gt_rounds.py: design, strata, quotas, severity-bucket filling, shared items, key id forms,
scoring, inter-rater agreement and adjudication (the server calls are thin and run by hand)."""
import doctest
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import gt_rounds as gr
from analysis import ai_disagree_relabel as rl

DESIGN = {"raters": {"A": "sidewalk_a", "B": "sidewalk_b"},
          "per_type": {"CurbRamp": 10, "Crosswalk": 10, "NoCurbRamp": 10, "Obstacle": 10, "SurfaceProblem": 10},
          "shares": {"margin": 0.5, "lone": 0.3, "light_voted": 0.2}, "margin_correct_cap": 0.4,
          "decoys_per_type": 4, "shared": 6}


def _pool(rows):
    base = {"schema": "sidewalk_a", "label_type": "CurbRamp", "severity": 1, "tags": np.nan, "ai_vote": "Disagree",
            "ai_accuracy": 0.95, "h_agree": 0, "h_dis": 0, "h_unsure": 0, "expert1": np.nan, "expert2": np.nan,
            "expert1_labeled": "f", "expert2_labeled": "f", "pano_id": "p", "heading": 0.0, "pitch": 0.0}
    df = pd.DataFrame([{**base, "label_id": i + 1, **r} for i, r in enumerate(rows)])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return rl.load_pool(buf)


def _city(schema, n_per_stratum=30, types=gr.LABEL_TYPES, seed=0):
    """A city with ``n_per_stratum`` rows of every stratum and type, severities cycling 1..5 and some missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in types:
        for s in gr.STRATA:
            for i in range(n_per_stratum):
                r = {"schema": schema, "label_type": t, "severity": [1, 2, 3, 4, 5, np.nan][i % 6]}
                if s == "lone":
                    r.update(ai_vote="Disagree")
                elif s == "light_voted":
                    r.update(ai_vote="Disagree", h_agree=2, h_dis=1)
                elif s == "margin_incorrect":
                    r.update(ai_vote="Disagree", h_agree=0, h_dis=2 + int(rng.integers(0, 3)))
                elif s == "margin_correct":
                    r.update(ai_vote="Disagree", h_agree=3 + int(rng.integers(0, 3)), h_dis=0)
                elif s == "decoy_agree":
                    r.update(ai_vote="Agree", h_agree=1)
                else:
                    r.update(ai_vote="Unsure")
                rows.append(r)
    return rows


def test_doctests():
    assert doctest.testmod(gr).failed == 0


def test_design_file_in_repo_is_valid_and_sums_to_250_per_city():
    d = gr.load_design(os.path.join(os.path.dirname(gr.__file__), "gt_rounds", "round2.json"))
    q = gr.quotas(d)
    assert sum(q.values()) == 250
    assert sum(n for (c, t), n in q.items() if c != "decoy") == 170
    assert len(d["raters"]) == 2 and len(set(d["raters"].values())) == 2
    assert not any(k.lower().endswith("_id") for k in d)  # no rater ids in the design


def test_eligible_excludes_every_experts_votes_and_placements():
    p = _pool([{}, {"expert1": "Agree"}, {"expert2": "Disagree"}, {"expert1_labeled": "t"},
               {"expert2_labeled": "true"}, {"label_type": "Signal"}])
    assert list(gr.eligible(p).label_id) == [1]


def test_eligible_without_expert2_columns_still_works():
    p = _pool([{}, {"expert1": "Agree"}]).drop(columns=["expert2", "expert2_labeled"])
    assert list(gr.eligible(p).label_id) == [1]


def test_eligible_drops_excluded_authors_and_tolerates_older_pools():
    # Gallery never shows an excluded author's labels, so a rater could not vote on them (round 2: 11 Seattle items).
    p = _pool([{}, {}, {}]).assign(author_excluded=["f", "t", "true"])
    assert list(gr.eligible(p).label_id) == [1]
    assert len(gr.eligible(p.drop(columns=["author_excluded"]))) == 3  # a pool pulled before the column existed


def test_strata_follow_the_margin_rule_of_data_generation():
    # The rule of data_generation.eligible_labels: incorrect at net Disagree > 1, correct at net Agree > 2.
    p = _pool([{"h_agree": 0, "h_dis": 2}, {"h_agree": 3, "h_dis": 0}, {"h_agree": 2, "h_dis": 0},
               {"h_agree": 1, "h_dis": 2}, {"h_agree": 2, "h_dis": 1}, {"h_agree": 4, "h_dis": 1},
               {"h_agree": 1, "h_dis": 1}, {"h_agree": 1, "h_dis": 0}])
    assert list(gr.margin_class(p)) == ["incorrect", "correct", "", "", "", "correct", "", ""]
    assert list(gr.assign_stratum(p)) == ["margin_incorrect", "margin_correct", "light_voted", "light_voted",
                                          "light_voted", "margin_correct", "", ""]  # one decisive vote: no stratum


def test_prepare_and_candidates_cap_margin_correct_and_rank_agree_decoys_first():
    p = _pool(_city("sidewalk_a") + _city("sidewalk_b"))
    prep = gr.prepare(p, "sidewalk_a")
    assert set(prep.schema) == {"sidewalk_a"}
    assert set(prep.cell) == set(gr.CELLS)
    c = gr.candidates(prep, "seed", DESIGN, oversample=1)
    q = gr.quotas(DESIGN)
    for (cell, t), n in q.items():
        g = c[(c.cell == cell) & (c.label_type == t)]
        if cell == "margin":
            assert (g.stratum == "margin_correct").sum() == int(np.floor(0.4 * n))
            assert (g.stratum == "margin_incorrect").sum() == n
            assert list(g.prio) == sorted(g.prio)
        else:
            assert len(g) == n
        if cell == "decoy":
            assert set(g.stratum) == {"decoy_agree"}  # Agree first, Unsure only as a top-up


def test_candidates_are_deterministic_in_the_seed_and_change_with_it():
    p = _pool(_city("sidewalk_a"))
    prep = gr.prepare(p, "sidewalk_a")
    a, b = gr.candidates(prep, "s1", DESIGN), gr.candidates(prep, "s1", DESIGN)
    assert list(a.label_uid) == list(b.label_uid)
    assert list(gr.candidates(prep, "s2", DESIGN).label_uid) != list(a.label_uid)


def test_pick_fills_severity_buckets_in_pool_proportion_and_spills():
    p = _pool(_city("sidewalk_a"))
    prep = gr.prepare(p, "sidewalk_a")
    shares = gr.bucket_shares(prep)
    c = gr.candidates(prep, "s", DESIGN)
    picked, short = gr.pick(c, set(c.label_uid), DESIGN, shares)
    assert not short
    assert len(picked) == sum(gr.quotas(DESIGN).values())
    g = picked[(picked.cell == "lone") & (picked.label_type == "CurbRamp")]
    want = gr.largest_remainder(len(g), shares[("lone", "CurbRamp")])
    assert g.sev_bucket.value_counts().to_dict() == {b: k for b, k in want.items() if k}
    mg = picked[(picked.cell == "margin") & (picked.label_type == "CurbRamp")]
    assert (mg.stratum == "margin_correct").sum() == 2 and len(mg) == 5
    # a bucket with no available rows spills into the others
    avail = set(c[c.sev_bucket != "none"].label_uid)
    picked2, short2 = gr.pick(c, avail, DESIGN, shares)
    assert not short2 and "none" not in set(picked2[picked2.cell != "decoy"].sev_bucket)


def test_pick_reports_shortfall_per_cell_and_tops_margin_up_with_incorrect():
    p = _pool(_city("sidewalk_a", n_per_stratum=2))
    prep = gr.prepare(p, "sidewalk_a")
    c = gr.candidates(prep, "s", DESIGN)
    picked, short = gr.pick(c, set(c.label_uid), DESIGN, gr.bucket_shares(prep))
    # quotas per type: lone 3, light_voted 2, margin 5, decoy 4; the city has 2 of each stratum
    assert short == {f"{cell}/{t}": 1 for t in gr.LABEL_TYPES for cell in ("lone", "margin")}
    mg = picked[(picked.cell == "margin") & (picked.label_type == "CurbRamp")]
    assert mg.stratum.value_counts().to_dict() == {"margin_correct": 2, "margin_incorrect": 2}
    # with no margin_correct rows at all, margin_incorrect fills the whole cell
    avail = set(c[c.stratum != "margin_correct"].label_uid)
    big = _pool(_city("sidewalk_a"))
    prep2 = gr.prepare(big, "sidewalk_a")
    c2 = gr.candidates(prep2, "s", DESIGN)
    picked2, short2 = gr.pick(c2, set(c2[c2.stratum != "margin_correct"].label_uid), DESIGN, gr.bucket_shares(prep2))
    assert not short2 and set(picked2[picked2.cell == "margin"].stratum) == {"margin_incorrect"}


def test_mark_shared_is_seeded_and_counts_exactly():
    p = _pool(_city("sidewalk_a"))
    prep = gr.prepare(p, "sidewalk_a")
    c = gr.candidates(prep, "s", DESIGN)
    picked, _ = gr.pick(c, set(c.label_uid), DESIGN, gr.bucket_shares(prep))
    a = gr.mark_shared(picked, "x", 6)
    assert int(a.shared.sum()) == 6
    assert list(a.shared) == list(gr.mark_shared(picked, "x", 6).shared)
    assert list(a.shared) != list(gr.mark_shared(picked, "y", 6).shared)
    assert a[a.shared].cell.nunique() > 1


def test_rater_items_own_city_first_then_other_cities_shared():
    key = pd.DataFrame({"packet_id": list("abcd"), "rater": ["A", "A", "B", "B"], "shared": [True, False, True, False]})
    assert list(gr.rater_items(key, "A").packet_id) == ["a", "b", "c"]
    assert list(gr.rater_items(key, "B").packet_id) == ["c", "d", "a"]


def test_round3_splits_and_city_ids(tmp_path):
    for t in gr.LABEL_TYPES:
        os.makedirs(tmp_path / t)
        pd.DataFrame({"label_uid": [f"a-xx:1", f"b-yy:2"], "split": ["train", "test"], "x": 0}).to_csv(
            tmp_path / t / "split.csv", index=False)
    splits, shas = gr.round3_splits(str(tmp_path))
    assert splits == {"a-xx:1": "train", "b-yy:2": "test"} and set(shas) == set(gr.LABEL_TYPES)
    conf = tmp_path / "cityparams.conf"
    conf.write_text('city-params {\n  db-schema {\n    a-xx = "sidewalk_a"\n  }\n  landing-page-url {\n    prod {\n'
                    '      a-xx = "https://a"\n    }\n  }\n}\n')
    assert gr.city_ids(["sidewalk_a"], str(conf)) == {"sidewalk_a": "a-xx"}
    with pytest.raises(ValueError, match="sidewalk_b"):
        gr.city_ids(["sidewalk_a", "sidewalk_b"], str(conf))


def _rated(rows):
    base = {"stratum": "lone", "label_type": "CurbRamp", "ai_vote": "Disagree", "margin_class": "", "verdict": "",
            "rater": "A", "shared": False}
    return pd.DataFrame([{**base, "packet_id": f"r{i}", **r} for i, r in enumerate(rows)])


def test_score_table_confirms_ai_and_training_target_on_margin_rows():
    m = _rated([{"verdict": "incorrect"}, {"verdict": "correct"}, {"verdict": "cant_tell"},
                {"stratum": "margin_incorrect", "margin_class": "incorrect", "verdict": "incorrect"},
                {"stratum": "margin_incorrect", "margin_class": "incorrect", "verdict": "correct"},
                {"stratum": "margin_correct", "margin_class": "correct", "verdict": "correct"},
                {"stratum": "decoy_agree", "ai_vote": "Agree", "verdict": "correct"}])
    t = gr.score_table(m).set_index(["stratum", "label_type"])
    assert t.loc[("lone", "all"), "n"] == 2 and t.loc[("lone", "all"), "rate"] == 0.5
    assert t.loc[("lone", "all"), "cant_tell"] == 1
    assert t.loc[("margin_incorrect", "all"), "target_rate"] == 0.5 and t.loc[("margin_incorrect", "all"), "rate"] == 0.5
    assert t.loc[("margin_correct", "all"), "rate"] == 0.0 and t.loc[("margin_correct", "all"), "target_rate"] == 1.0
    assert t.loc[("decoy_agree", "all"), "rate"] == 1.0
    m2 = pd.concat([m, _rated([{"verdict": ""}])], ignore_index=True)
    assert gr.score_table(m2).set_index(["stratum", "label_type"]).loc[("lone", "all"), "unanswered"] == 1
    w = gr.score_table(m, {"lone": {"CurbRamp": 100}, "margin_incorrect": {"CurbRamp": 100}}).set_index(
        ["stratum", "label_type"])
    assert w.loc[("lone", "all"), "w_rate"] == 0.5 and w.loc[("margin_incorrect", "all"), "w_rate"] == 0.5
    m3 = m.copy()
    m3.loc[m3.stratum == "margin_incorrect", "verdict"] = "incorrect"
    w3 = gr.score_table(m3, {"lone": {"CurbRamp": 100}, "margin_incorrect": {"CurbRamp": 100}}).set_index(
        ["stratum", "label_type"])
    assert w3.loc[("lone", "all"), "w_rate"] == 0.5 and w3.loc[("margin_incorrect", "all"), "w_rate"] == 1.0
    md = gr.score_markdown(gr.score_table(m, {"lone": {"CurbRamp": 100}}))
    assert "population-weighted" in md and "training target" in md
    assert "r0" not in md


def test_agreement_table_percent_kappa_and_each_rater_vs_ai():
    key = _rated([{"shared": True}, {"shared": True}, {"shared": True, "ai_vote": "Agree", "stratum": "decoy_agree"},
                  {"shared": False}, {"shared": True}])
    sa = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "correct", "correct", "incorrect", "cant_tell"]})
    sb = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "incorrect", "correct", "correct", "correct"]})
    t = gr.agreement_table(key, {"A": sa, "B": sb}).set_index("label_type")
    assert t.loc["all", "n"] == 3 and t.loc["all", "shared_items"] == 4
    assert t.loc["all", "agree"] == pytest.approx(2 / 3)
    assert t.loc["all", "n_disagree"] == 2
    assert t.loc["all", "A_confirms_ai"] == 0.5 and t.loc["all", "B_confirms_ai"] == 1.0
    assert not np.isnan(t.loc["all", "kappa"])
    assert t.loc["all", "n_answered"] == 4 and t.loc["all", "agree3"] == pytest.approx(2 / 4)
    assert t.loc["all", "cant_tell_splits"] == 1
    with pytest.raises(ValueError):
        gr.agreement_table(key, {"A": sa})


def test_agreement_needs_both_decided_counts_per_type_and_leaves_unmasked_out():
    key = _rated([{"shared": True}, {"shared": True}, {"shared": True, "label_type": "Obstacle"},
                  {"shared": True, "label_type": "Obstacle"}])
    sa = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "correct", "incorrect", "correct"]})
    sb = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "cant_tell", "incorrect", "incorrect"]})
    t = gr.agreement_table(key, {"A": sa, "B": sb}).set_index("label_type")
    assert t.loc["all", "n"] == 3 and t.loc["CurbRamp", "n"] == 1  # B's can't tell is not decided
    assert t.loc["CurbRamp", "shared_items"] == 2 and t.loc["Obstacle", "shared_items"] == 2
    u = gr.agreement_table(key, {"A": sa, "B": sb}, unmasked={"r0"}).set_index("label_type")
    assert u.loc["all", "shared_items"] == 3 and u.loc["all", "unmasked_excluded"] == 1
    assert u.loc["CurbRamp", "unmasked_excluded"] == 1 and u.loc["Obstacle", "unmasked_excluded"] == 0


@pytest.mark.parametrize("bad", ["Correct", "correct ", "agree"])
def test_shared_pairs_refuses_unknown_verdicts(bad):
    key = _rated([{"shared": True}, {"shared": True}])
    sa = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["correct", bad]})
    sb = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["correct", "correct"]})
    with pytest.raises(ValueError, match="unexpected verdicts"):
        gr.shared_pairs(key, {"A": sa, "B": sb})


def test_shared_pairs_refuses_short_repeated_or_foreign_sheets():
    key = _rated([{"shared": True}, {"shared": True}])
    good = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["correct", "correct"]})
    for bad in (good.iloc[:1], pd.concat([good, good.iloc[:1]]),
                pd.concat([good, pd.DataFrame({"packet_id": ["zz"], "verdict": ["correct"]})])):
        with pytest.raises(ValueError):
            gr.shared_pairs(key, {"A": good, "B": bad})


def test_gallery_index_carries_rule_mask_and_shared_marker():
    page = gr.gallery_index_html("A", [("a-xx", "https://a", [1, 2], False), ("b-yy", "https://b", [3], True)], 3)
    assert gr.RATING_RULE_HTML in page and rl.GALLERY_MASK_CSS.splitlines()[0] in page
    assert "shared with the other rater" in page and "labelIds=1,2" in page


def test_key_columns_cover_both_id_forms_and_the_strata():
    for c in ("label_uid", "label_uid_city", "city_id", "stratum", "margin_class", "round3_split", "shared", "rater",
              "sev_bucket"):
        assert c in gr.KEY_COLUMNS


def test_build_parser_subcommands():
    p = gr.build_parser()
    for cmd in ("build", "gallery", "votes", "score", "agreement", "comments", "adjudicate-build",
                "adjudicate-report"):
        assert cmd in p.format_help()


def _adj_case():
    """Six shared items and one own-city item: two matches, a conflict, two can't-tell splits, both can't tell."""
    key = _rated([{"shared": True}, {"shared": True, "ai_vote": "Agree", "stratum": "decoy_agree"},
                  {"shared": True, "label_type": "Obstacle"}, {"shared": True, "label_type": "Obstacle"},
                  {"shared": True}, {"shared": True}, {"shared": False}])
    key["label_uid"] = [f"sidewalk_a:{i}" for i in range(len(key))]
    sa = pd.DataFrame({"packet_id": key.packet_id,
                       "verdict": ["incorrect", "correct", "correct", "cant_tell", "correct", "cant_tell", "correct"]})
    sb = pd.DataFrame({"packet_id": key.packet_id,
                       "verdict": ["incorrect", "correct", "incorrect", "incorrect", "cant_tell", "cant_tell", ""]})
    return key, {"A": sa, "B": sb}


def test_mismatches_list_every_differing_shared_item_in_type_order():
    key, sheets = _adj_case()
    mm = gr.mismatches(key, sheets)
    assert list(mm.packet_id) == ["r4", "r2", "r3"]  # CurbRamp before Obstacle, then packet id
    assert list(mm.order) == [1, 2, 3]
    assert dict(zip(mm.packet_id, mm.mismatch_kind)) == {"r2": "conflict", "r3": "cant_tell_split",
                                                         "r4": "cant_tell_split"}
    assert "r5" not in set(mm.packet_id) and "r6" not in set(mm.packet_id)  # both can't tell match; own-city item out


def test_mismatches_refuse_an_unfinished_shared_item():
    key, sheets = _adj_case()
    sheets["B"].loc[0, "verdict"] = ""
    with pytest.raises(ValueError, match="unanswered"):
        gr.mismatches(key, sheets)


def test_adjudicated_gold_comes_from_matches_then_consensus():
    key, sheets = _adj_case()
    cons = pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["correct", "no_consensus", "cant_tell"]})
    g = gr.adjudicated(key, sheets, cons).set_index("packet_id")
    assert g.loc["r0", "gold"] == "incorrect" and g.loc["r0", "source"] == "matched"
    assert g.loc["r5", "gold"] == "" and g.loc["r5", "source"] == "both_cant_tell"
    assert g.loc["r2", "gold"] == "correct" and g.loc["r2", "source"] == "consensus"
    assert g.loc["r3", "source"] == "no_consensus" and g.loc["r4", "source"] == "consensus_cant_tell"
    assert g.loc["r3", "gold"] == "" and g.loc["r4", "gold"] == ""
    assert "r6" not in g.index


def test_adjudicated_refuses_a_consensus_sheet_that_is_short_extra_or_blank():
    key, sheets = _adj_case()
    for cons in (pd.DataFrame({"packet_id": ["r2", "r3"], "consensus": ["correct", "correct"]}),
                 pd.DataFrame({"packet_id": ["r2", "r3", "r4", "r0"], "consensus": ["correct"] * 4}),
                 pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["correct", "", "correct"]}),
                 pd.DataFrame({"packet_id": ["r2", "r3", "r4", "r4"], "consensus": ["correct"] * 4})):
        with pytest.raises(ValueError):
            gr.adjudicated(key, sheets, cons)


def test_resolution_and_gold_tables():
    key, sheets = _adj_case()
    cons = pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["correct", "incorrect", "incorrect"]})
    g = gr.adjudicated(key, sheets, cons)
    r = gr.resolution_table(g, ["A", "B"]).set_index(["mismatch_kind", "label_type"])
    assert r.loc[("all", "all"), "n"] == 3
    assert r.loc[("all", "all"), "kept_A"] == 1 and r.loc[("all", "all"), "kept_B"] == 1  # r2 to A; r3 to B
    assert r.loc[("all", "all"), "new"] == 1  # r4: A correct, B can't tell, consensus incorrect
    assert r.loc[("conflict", "all"), "n"] == 1
    t = gr.gold_table(g).set_index("label_type")
    assert t.loc["all", "shared_items"] == 6 and t.loc["all", "gold"] == 5 and t.loc["all", "no_gold"] == 1
    assert t.loc["all", "from_matched"] == 2 and t.loc["all", "from_consensus"] == 3
    # AI Disagree gold: r0 incorrect, r2 correct, r3 incorrect, r4 incorrect -> 3 of 4 confirm the AI
    assert t.loc["all", "n_disagree"] == 4 and t.loc["all", "confirms_ai"] == pytest.approx(0.75)
    assert t.loc["all", "agree_correct"] == 1.0


def test_adjudication_page_hides_the_ai_and_crowd_and_carries_both_verdicts():
    key, sheets = _adj_case()
    key = key.assign(pano_id="p", heading=0.0, pitch=0.0, h_agree=7)
    mm = gr.mismatches(key, sheets)
    page = gr.adjudication_page(mm, ["A", "B"], {"A": {"sidewalk_a:2": "ramp is there"}}, "../packet/items", "t")
    assert "ramp is there" in page and "no_consensus" in page and "../packet/items/r2.jpg" in page
    for hidden in ("ai_vote", "stratum", "decoy_agree", "h_agree", "margin_class"):  # the rule text says Disagree
        assert hidden not in page
    assert gr.adjudication_columns(["A", "B"]) == ["packet_id", "order", "label_type", "mismatch_kind",
                                                   "verdict_A", "verdict_B", "consensus", "rationale",
                                                   "rule_clarification"]


def test_resolution_columns_partition_n_with_cant_tell_and_no_consensus_outcomes():
    key, sheets = _adj_case()
    cons = pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["incorrect", "cant_tell", "no_consensus"]})
    g = gr.adjudicated(key, sheets, cons)
    r = gr.resolution_table(g, ["A", "B"]).set_index(["mismatch_kind", "label_type"])
    row = r.loc[("all", "all")]
    assert (row.n, row.kept_A, row.kept_B, row.new, row.cant_tell, row.no_consensus) == (3, 0, 1, 0, 1, 1)
    # r3: A said can't tell, consensus can't tell -> counted once, under cant_tell, not under kept_A
    assert r.loc[("cant_tell_split", "all"), "kept_A"] == 0 and r.loc[("cant_tell_split", "all"), "cant_tell"] == 1
    t = gr.gold_table(g).set_index("label_type")
    assert t.loc["all", "from_consensus"] == 1 and t.loc["all", "no_gold"] == 3


def test_gold_table_by_stratum_and_unmasked_left_out():
    key, sheets = _adj_case()
    cons = pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["correct", "incorrect", "incorrect"]})
    g = gr.adjudicated(key, sheets, cons, unmasked={"r0"})
    assert g.set_index("packet_id").loc["r0", "gold"] == "incorrect"  # kept in the gold CSV, flagged
    t = gr.gold_table(g).set_index("label_type")
    assert t.loc["all", "shared_items"] == 5 and t.loc["all", "n_disagree"] == 3
    s = gr.gold_table(g, by="stratum").set_index("stratum")
    assert s.loc["decoy_agree", "agree_correct"] == 1.0 and s.loc["lone", "n_disagree"] == 3
    r = gr.resolution_table(gr.adjudicated(key, sheets, cons, unmasked={"r2"}), ["A", "B"])
    assert r.set_index(["mismatch_kind", "label_type"]).loc[("all", "all"), "n"] == 2
