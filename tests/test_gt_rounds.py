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
                "adjudicate-consensus", "adjudicate-report"):
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


def test_adjudication_columns():
    assert gr.adjudication_columns(["A", "B"]) == ["packet_id", "order", "label_type", "mismatch_kind",
                                                   "verdict_A", "verdict_B", "consensus", "rationale",
                                                   "rule_clarification"]


def _comments(rows):
    return pd.DataFrame(rows, columns=["label_uid", "timestamp", "comment"])


def test_consensus_from_votes_takes_matching_post_verdicts_and_new_or_edited_comments():
    items = pd.DataFrame({"packet_id": ["r1", "r2", "r3"], "label_uid": ["s:1", "s:2", "s:3"], "consensus": "",
                          "rationale": "", "rule_clarification": ""})
    post = {"A": pd.DataFrame({"packet_id": ["r3", "r1", "r2", "r9"],
                               "verdict": ["cant_tell", "correct", "correct", "incorrect"]}),
            "B": pd.DataFrame({"packet_id": ["r1", "r2", "r3"], "verdict": ["correct", "incorrect", "cant_tell"]})}
    pre = {"A": _comments([("s:1", 1, "old"), ("s:2", 1, "kept as is")])}
    after = {"A": _comments([("s:1", 5, "old. Update: agreed"), ("s:2", 1, "kept as is")]),
             "B": _comments([("s:1", 6, "fine"), ("s:3", 7, "cannot see it")])}
    c = gr.consensus_from_votes(items, post, pre, after).set_index("packet_id")
    assert c.consensus.tolist() == ["correct", "no_consensus", "cant_tell"]
    assert c.loc["r1", "rationale"] == "A: old. Update: agreed || A: [replaced; was: old] || B: fine"
    assert c.loc["r2", "rationale"] == "" and c.loc["r3", "rationale"] == "B: cannot see it"
    assert list(c.reset_index().columns) == list(items.columns)


def test_consensus_from_votes_reports_removed_comments_and_orders_new_ones_in_utc():
    items = pd.DataFrame({"packet_id": ["r1", "r2"], "label_uid": ["s:1", "s:2"], "consensus": "", "rationale": "",
                          "rule_clarification": ""})
    post = {r: pd.DataFrame({"packet_id": ["r1", "r2"], "verdict": ["correct", "correct"]}) for r in ("A", "B")}
    pre = {"A": _comments([("s:2", "2026-10-30 10:00:00-07", "ample space to pass")]), "B": _comments([])}
    # the hour DST ended: 01:10-08 is 09:10 UTC, after 01:30-07 (08:30 UTC), though it sorts first as a string
    after = {"A": _comments([("s:1", "2026-11-01 01:10:00-08", "second"), ("s:1", "2026-11-01 01:30:00-07", "first")]),
             "B": _comments([])}
    c = gr.consensus_from_votes(items, post, pre, after).set_index("packet_id")
    assert c.loc["r1", "rationale"] == "A: first || A: second"
    assert c.loc["r2", "rationale"] == "A: [removed; was: ample space to pass]"


def test_consensus_from_votes_refuses_a_missing_or_blank_post_vote():
    items = pd.DataFrame({"packet_id": ["r1", "r2"], "label_uid": ["s:1", "s:2"]})
    full = pd.DataFrame({"packet_id": ["r1", "r2"], "verdict": ["correct", "correct"]})
    for bad in (pd.DataFrame({"packet_id": ["r1"], "verdict": ["correct"]}),
                pd.DataFrame({"packet_id": ["r1", "r2"], "verdict": ["correct", ""]}),
                pd.DataFrame({"packet_id": ["r1", "r2"], "verdict": ["correct", "maybe"]})):
        with pytest.raises(ValueError):
            gr.consensus_from_votes(items, {"A": full, "B": bad})


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


def test_agreement_three_answer_view_skips_one_sided_blanks_and_counts_splits_in_kappa():
    key = _rated([{"shared": True}, {"shared": True}, {"shared": True}, {"shared": True}, {"shared": True}])
    sa = pd.DataFrame({"packet_id": key.packet_id,
                       "verdict": ["incorrect", "correct", "cant_tell", "incorrect", "correct"]})
    sb = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "correct", "correct", "", "correct"]})
    t = gr.agreement_table(key, {"A": sa, "B": sb}).set_index("label_type")
    assert t.loc["all", "n_answered"] == 4 and t.loc["all", "agree3"] == pytest.approx(3 / 4)
    assert t.loc["all", "kappa3"] != pytest.approx(t.loc["all", "kappa"])  # the split lowers the 3-answer kappa
    assert t.loc["all", "A_confirms_ai"] == pytest.approx(1 / 3) and t.loc["all", "B_confirms_ai"] == pytest.approx(1 / 3)


def test_gold_rates_count_unsure_decoys_apart_from_disagrees():
    gl = pd.DataFrame({"ai_vote": ["Disagree", "Unsure", "Agree"], "gold": ["incorrect", "incorrect", "correct"]})
    r = gr._gold_rates(gl)
    assert r["n_disagree"] == 1 and r["confirms_ai"] == 1.0 and r["n_unsure"] == 1 and r["n_agree"] == 1


def test_unmasked_ids_are_checked_against_the_key():
    key = pd.DataFrame({"packet_id": ["r0", "r1"], "label_uid": ["sidewalk_a:1", "sidewalk_a:2"]})
    assert gr.unmasked_ids(pd.DataFrame({"packet_id": ["r1"], "label_uid": ["sidewalk_a:2"]}), key) == {"r1"}
    for bad in (pd.DataFrame({"packet_id": ["zz"], "label_uid": ["sidewalk_a:2"]}),
                pd.DataFrame({"packet_id": ["r1"], "label_uid": ["sidewalk_b:2"]}),
                pd.DataFrame({"packet_id": ["r1", "r1"], "label_uid": ["sidewalk_a:2"] * 2})):
        with pytest.raises(ValueError):
            gr.unmasked_ids(bad, key)


def _cli_round(tmp_path):
    """A pinned key, pool, two pinned sheets and a pinned unmasked list on disk, as the commands expect."""
    key, sheets = _adj_case()
    key = key.assign(label_uid_city=key.label_uid, rater=["A"] * 6 + ["B"], margin_class="", round3_split="",
                     pano_id="p", schema="sidewalk_a", label_id=range(len(key))).drop(columns="verdict")
    kdir = tmp_path / "key"
    kdir.mkdir()
    kp, pp = kdir / "key.csv", tmp_path / "pool.csv"
    key.to_csv(kp, index=False)
    pd.DataFrame({"schema": "sidewalk_a", "label_id": range(len(key)), "heading": 0.0, "pitch": 0.0, "h_agree": 0,
                  "h_dis": 0}).to_csv(pp, index=False)
    mp = kdir / "manifest.json"
    rl.write_manifest(str(mp), {"design": "t", "key_sha256": rl.sha256_file(kp), "pool_sha256": rl.sha256_file(pp)})
    paths = {}
    for r, s in sheets.items():
        paths[r] = tmp_path / f"votes_{r}.csv"
        s.assign(label_type="CurbRamp", notes="").to_csv(paths[r], index=False)
        rl.pin(str(mp), f"votes_{r}", str(paths[r]), len(s))
    up = tmp_path / "unmasked.csv"
    pd.DataFrame({"packet_id": ["r0"], "label_uid": ["sidewalk_a:0"], "reason": "test"}).to_csv(up, index=False)
    rl.pin(str(mp), "unmasked", str(up), 1)
    base = ["--key", str(kp), "--sheets", f"A={paths['A']}", f"B={paths['B']}"]
    return base, mp, pp, up


def test_cli_build_report_and_refusals(tmp_path, capsys):
    base, mp, pp, up = _cli_round(tmp_path)
    out = tmp_path / "adj"
    build = ["adjudicate-build", *base, "--out", str(out)]
    with pytest.raises(SystemExit, match="--unmasked"):  # the manifest pins an unmasked list
        gr.main(build)
    gr.main(build + ["--unmasked", str(up)])
    assert "adjudication_items" in json.load(open(mp))["rating"]
    listed = (out / "items.csv").read_bytes()
    with pytest.raises(SystemExit):  # a rerun refuses and leaves the list alone
        gr.main(build + ["--unmasked", str(up)])
    assert (out / "items.csv").read_bytes() == listed
    items = pd.read_csv(out / "items.csv", dtype=str, keep_default_na=False)
    listed = set(items.packet_id)
    # the meeting: both raters end on correct on the listed items in prod; the post-meeting pulls get their own pins
    for r in ("A", "B"):
        path = tmp_path / f"post_{r}.csv"
        pre = pd.read_csv(tmp_path / f"votes_{r}.csv", dtype=str, keep_default_na=False)
        pre.assign(verdict=pre.verdict.where(~pre.packet_id.isin(listed), "correct")).to_csv(path, index=False)
        rl.pin(str(mp), f"votes_{r}_post", str(path), 7)
        for stage, rows in (("", [("sidewalk_a", 2, "2026-10-30 10:00:00-07", "", f"{r} first look")]),
                            ("_post", [("sidewalk_a", 2, "2026-11-02 13:50:00-08", "", f"{r} agreed")])):
            cpath = tmp_path / f"comments_{r}{stage}.csv"
            pd.DataFrame(rows).to_csv(cpath, index=False, header=False)
            rl.pin(str(mp), f"comments_{r}{stage}", str(cpath), 1)
    consensus = ["adjudicate-consensus", "--key", base[1], "--items", str(out / "items.csv"), *base[2:],
                 "--comments", f"A={tmp_path / 'comments_A.csv'}", f"B={tmp_path / 'comments_B.csv'}",
                 "--post-comments", f"A={tmp_path / 'comments_A_post.csv'}", f"B={tmp_path / 'comments_B_post.csv'}",
                 "--post-sheets", f"A={tmp_path / 'post_A.csv'}", f"B={tmp_path / 'post_B.csv'}"]
    with pytest.raises(SystemExit, match="not the pinned"):  # an unpinned post-meeting sheet
        gr.main(consensus[:-2] + [f"A={tmp_path / 'post_A.csv'}", f"B={tmp_path / 'votes_B.csv'}",
                                  "--out", str(out / "x.csv")])
    swapped = [a.replace("B=", "C=", 1) if a.startswith("B=") and "comments_B.csv" in a else a for a in consensus]
    with pytest.raises(SystemExit, match="names raters"):  # comment pulls must name the same raters
        gr.main(swapped + ["--out", str(out / "x.csv")])
    with pytest.raises(SystemExit):  # the comment pulls are not optional
        gr.main(consensus[:consensus.index("--comments")] + consensus[consensus.index("--post-sheets"):]
                + ["--out", str(out / "x.csv")])
    assert "--comments" in capsys.readouterr().err
    # a vote changed after the meeting on an item that is not on the list is refused
    moved = tmp_path / "post_A_moved.csv"
    post_a = pd.read_csv(tmp_path / "post_A.csv", dtype=str, keep_default_na=False)
    unlisted = next(p for p in post_a.packet_id if p not in listed)
    post_a.assign(verdict=post_a.verdict.where(post_a.packet_id != unlisted, "cant_tell")).to_csv(moved, index=False)
    rl.pin(str(mp), "votes_A_post", str(moved), 7, repin=True)
    with pytest.raises(SystemExit, match=f"not on the list.*A:{unlisted}"):
        gr.main(consensus[:-2] + [f"A={moved}", f"B={tmp_path / 'post_B.csv'}", "--out", str(out / "x.csv")])
    assert not (out / "x.csv").exists()
    rl.pin(str(mp), "votes_A_post", str(tmp_path / "post_A.csv"), 7, repin=True)
    gr.main(consensus + ["--out", str(out / "consensus.csv")])
    assert "adjudication_consensus" in json.load(open(mp))["rating"]  # adjudicate-consensus pins its output
    cons = pd.read_csv(out / "consensus.csv", dtype=str, keep_default_na=False)
    assert set(cons.consensus) == {"correct"} and list(cons.columns) == list(items.columns)
    r2 = cons.set_index("packet_id").rationale
    assert any(v == "A: A agreed || A: [replaced; was: A first look] || B: B agreed || B: [replaced; was: B first look]"
               for v in r2)
    report = ["adjudicate-report", *base, "--unmasked", str(up), "--items", str(out / "items.csv")]
    # a changed fixed column, and a changed consensus verdict, both fail the consensus pin
    cons.assign(verdict_A="incorrect").to_csv(out / "tampered.csv", index=False)
    edited = cons.copy()
    edited.loc[0, "consensus"] = "cant_tell"
    edited.to_csv(out / "edited.csv", index=False)
    for bad in ("tampered.csv", "edited.csv"):
        with pytest.raises(SystemExit, match="not the pinned adjudication_consensus"):
            gr.main(report + ["--consensus", str(out / bad), "--out", str(tmp_path / "bad.csv")])
    assert not (tmp_path / "bad.csv").exists()
    gr.main(report + ["--consensus", str(out / "consensus.csv"), "--out", str(tmp_path / "gold.csv")])
    gold = pd.read_csv(tmp_path / "gold.csv", dtype=str, keep_default_na=False).set_index("packet_id")
    assert gold.loc["r0", "unmasked"] == "True" and gold.loc["r2", "gold"] == "correct"
    assert {"adjudication_consensus", "gold_shared"} <= set(json.load(open(mp))["rating"])
    assert "1 unmasked items left out" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # gold is not overwritten
        gr.main(report + ["--consensus", str(out / "consensus.csv"), "--out", str(tmp_path / "gold.csv")])


def test_cli_stage_takes_only_post(capsys):
    with pytest.raises(SystemExit):
        gr.build_parser().parse_args(["votes", "--key", "k", "--rater", "A", "--rater-id", "u", "--out", "o",
                                      "--stage", "Post"])
    assert "invalid choice" in capsys.readouterr().err
    args = gr.build_parser().parse_args(["comments", "--key", "k", "--rater", "A", "--rater-id", "u", "--out", "o",
                                         "--stage", "post"])
    assert gr._pin_name("comments", args.rater, args.stage) == "comments_A_post"


def test_unsampled_disagrees_partition_the_unstratified_disagrees():
    rows = [{"ai_vote": "Disagree", "h_agree": a, "h_dis": d, "h_unsure": u}
            for a, d, u in [(1, 0, 0), (0, 1, 3), (0, 0, 1), (1, 1, 0), (2, 2, 1), (0, 0, 0), (0, 2, 0), (2, 1, 0)]]
    df = pd.DataFrame(rows + [{"ai_vote": "Agree", "h_agree": 1, "h_dis": 0, "h_unsure": 0}])
    u = gr.unsampled_disagrees(df)
    assert u == {"one_decisive": 2, "unsure_only": 1, "tie": 2}
    assert sum(u.values()) == int(((df.ai_vote == "Disagree") & (gr.assign_stratum(df) == "")).sum())


def test_cli_preview_pins_nothing_and_writes_no_final_files(tmp_path):
    base, mp, pp, up = _cli_round(tmp_path)
    loose = tmp_path / "votes_B_unpinned.csv"
    pd.read_csv(tmp_path / "votes_B.csv").to_csv(loose, index=False)
    loose.write_text(loose.read_text() + "\n")  # different bytes, so not the pinned file
    before = mp.read_bytes()
    out = tmp_path / "adj"
    args = ["adjudicate-build", "--key", base[1], "--sheets", base[3], f"B={loose}", "--out", str(out),
            "--unmasked", str(up), "--allow-unpinned"]
    gr.main(args)
    assert mp.read_bytes() == before
    assert (out / "items.preview.csv").exists()
    assert not (out / "items.csv").exists() and not (out / "meeting_record.md").exists()


def test_cli_score_leaves_unmasked_items_out(tmp_path, capsys):
    base, mp, pp, up = _cli_round(tmp_path)
    gr.main(["score", "--key", base[1], "--rater", "B", "--sheet", base[4].split("=", 1)[1], "--unmasked", str(up)])
    assert "1 unmasked items left out" in capsys.readouterr().out
