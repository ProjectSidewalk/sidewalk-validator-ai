"""Tests for analysis/gt_rounds.py: design, strata, quotas, severity-bucket filling, shared items, key id forms,
scoring, inter-rater agreement and adjudication (the server calls are thin and run by hand)."""
import doctest
import io
import json
import os
import re

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


def test_mark_shared_draws_apart_from_the_candidate_ranking():
    """The shared draw ranks on ``<seed>:shared``, so with the same seed it is not simply the best-ranked candidates of
    each stratum (which would tie the shared set to the sampling order)."""
    p = _pool(_city("sidewalk_a"))
    prep = gr.prepare(p, "sidewalk_a")
    c = gr.candidates(prep, "s", DESIGN)
    picked, _ = gr.pick(c, set(c.label_uid), DESIGN, gr.bucket_shares(prep))
    a = gr.mark_shared(picked, "s", 6)

    def top(seed):
        out = set()
        for (cell, t), g in a.groupby(["cell", "label_type"]):
            k = int(g.shared.sum())
            out |= set(sorted(g.label_uid, key=lambda u: rl.seeded_rank(seed, u))[:k])
        return out

    assert set(a.label_uid[a.shared]) == top("s:shared")
    assert set(a.label_uid[a.shared]) != top("s")


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
    # min_cell=1: the 2-item cells here sit under the default minimum (5), covered by its own test below
    w = gr.score_table(m, {"lone": {"CurbRamp": 100}, "margin_incorrect": {"CurbRamp": 100}}, min_cell=1).set_index(
        ["stratum", "label_type"])
    assert w.loc[("lone", "all"), "w_rate"] == 0.5 and w.loc[("margin_incorrect", "all"), "w_rate"] == 0.5
    m3 = m.copy()
    m3.loc[m3.stratum == "margin_incorrect", "verdict"] = "incorrect"
    w3 = gr.score_table(m3, {"lone": {"CurbRamp": 100}, "margin_incorrect": {"CurbRamp": 100}}, min_cell=1).set_index(
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


def test_cant_tell_splits_leave_out_items_both_raters_could_not_tell():
    key, sheets = _adj_case()  # shared: two matches, a conflict, two can't-tell splits, one both can't tell
    t = gr.agreement_table(key, sheets).set_index("label_type")
    assert t.loc["all", "cant_tell_splits"] == 2
    assert t.loc["all", "n_answered"] == 6 and t.loc["all", "agree3"] == pytest.approx(3 / 6)


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
    pinned = mp.read_bytes()
    (out / "consensus.csv").rename(out / "consensus_first.csv")
    gr.main(consensus + ["--out", str(out / "consensus.csv")])  # the same content again: the manifest is untouched
    assert mp.read_bytes() == pinned
    (out / "consensus.csv").unlink()
    rl.pin(str(mp), "comments_A_post", str(tmp_path / "comments_A.csv"), 1, repin=True)
    with pytest.raises(SystemExit, match="already pinned"):  # different content needs --repin; nothing written
        gr.main([a.replace("comments_A_post.csv", "comments_A.csv") for a in consensus]
                + ["--out", str(out / "consensus.csv")])
    assert not (out / "consensus.csv").exists() and mp.read_bytes() != pinned
    rl.pin(str(mp), "comments_A_post", str(tmp_path / "comments_A_post.csv"), 1, repin=True)
    (out / "consensus_first.csv").rename(out / "consensus.csv")
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
    # gold pinned to other content: refused before the new gold file is written, and the manifest is untouched
    other = tmp_path / "other_gold.csv"
    other.write_text("packet_id,gold\nr2,incorrect\n")
    rl.pin(str(mp), "gold_shared", str(other), 1, repin=True)
    pinned = mp.read_bytes()
    with pytest.raises(SystemExit, match="gold_shared is already pinned"):
        gr.main(report + ["--consensus", str(out / "consensus.csv"), "--out", str(tmp_path / "gold2.csv")])
    assert not (tmp_path / "gold2.csv").exists() and mp.read_bytes() == pinned


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


# ---------------------------------------------------------------------------------------------------------------------
# #85: recount without excluded authors, cells under n=5, the opt-in thin_voted stratum
# ---------------------------------------------------------------------------------------------------------------------

THIN_DESIGN = {**DESIGN, "shares": {"margin": 0.4, "lone": 0.25, "light_voted": 0.15, "thin_voted": 0.2}}
ROUND2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "relabel-2026-10", "round2")


def _thin_rows(schema, n=4):
    """AI Disagrees in no round-2 stratum: one decisive vote, only Unsure, and a tie, ``n`` of each per type."""
    rows = []
    for t in gr.LABEL_TYPES:
        for votes in ({"h_agree": 1}, {"h_unsure": 2}, {"h_agree": 1, "h_dis": 1}):
            rows += [{"schema": schema, "label_type": t, "ai_vote": "Disagree", **votes}] * n
    return rows


def _population_by_hand(pool, schema, strata=gr.DISAGREE_STRATA, thin=False):
    e = gr.eligible(pool[pool.schema == schema])
    st = gr.assign_stratum(e, thin=thin)
    return {s: {t: int(((st == s) & (e.label_type == t)).sum()) for t in gr.LABEL_TYPES} for s in strata}


def test_recount_population_matches_build_on_a_pool_without_the_column():
    p = _pool(_city("sidewalk_a") + _city("sidewalk_b", seed=1) + _thin_rows("sidewalk_a"))
    r = gr.recount_population(p, DESIGN)
    assert r["population"]["A"] == _population_by_hand(p, "sidewalk_a")  # what cmd_build pinned before #85
    assert r["population"]["B"] == _population_by_hand(p, "sidewalk_b")
    e = gr.eligible(p[p.schema == "sidewalk_a"])
    assert r["eligible_disagrees"]["sidewalk_a"] == int((e.ai_vote == "Disagree").sum())
    assert r["unsampled"]["sidewalk_a"] == {"one_decisive": 20, "unsure_only": 20, "tie": 20}
    assert r["unsampled"]["sidewalk_b"] == {"one_decisive": 0, "unsure_only": 0, "tie": 0}


@pytest.mark.skipif(not os.path.exists(os.path.join(ROUND2, "pool.csv.gz")), reason="committed round-2 pool absent")
def test_recount_population_reproduces_the_committed_round2_manifest():
    design = gr.load_design(gr.DEFAULT_DESIGN)
    pool = rl.load_pool(os.path.join(ROUND2, "pool.csv.gz"))
    man = json.load(open(os.path.join(ROUND2, "key", "manifest.json")))
    r = gr.recount_population(pool, design)
    assert r["population"] == man["population"]
    # the doc's denominators (docs/ai-disagree-relabel-2026-10.md, Round 2), before the exclusion
    assert r["eligible_disagrees"] == {"sidewalk_seattle": 17284, "sidewalk_chicago": 10769}
    assert sum(r["unsampled"]["sidewalk_seattle"].values()) == 8611
    assert sum(r["unsampled"]["sidewalk_chicago"].values()) == 3853


def test_recount_drops_excluded_authors_from_weights_and_denominators():
    p = _pool(_city("sidewalk_a", n_per_stratum=6) + _city("sidewalk_b", n_per_stratum=6) + _thin_rows("sidewalk_a"))
    before = gr.recount_population(p, DESIGN)
    a = p[p.schema == "sidewalk_a"]
    lone = a[(a.label_type == "CurbRamp") & (gr.assign_stratum(a) == "lone")].label_id.iloc[0]
    light = a[(a.label_type == "Obstacle") & (gr.assign_stratum(a) == "light_voted")].label_id.iloc[0]
    tie = a[(a.ai_vote == "Disagree") & (a.h_agree == 1) & (a.h_dis == 1)].label_id.iloc[0]
    decoy = a[a.ai_vote == "Agree"].label_id.iloc[0]
    flags = pd.DataFrame({"schema": "sidewalk_a", "label_id": a.label_id,
                          "author_excluded": np.where(a.label_id.isin([lone, light, tie, decoy]), "t", "f")})
    after = gr.recount_population(gr.apply_author_excluded(p, flags)[0], DESIGN)
    pb, pa = before["population"]["A"], after["population"]["A"]
    assert pb["lone"]["CurbRamp"] - pa["lone"]["CurbRamp"] == 1
    assert pb["light_voted"]["Obstacle"] - pa["light_voted"]["Obstacle"] == 1
    moved = {(s, t) for s in pb for t in pb[s] if pb[s][t] != pa[s][t]}
    assert moved == {("lone", "CurbRamp"), ("light_voted", "Obstacle")}
    assert before["eligible_disagrees"]["sidewalk_a"] - after["eligible_disagrees"]["sidewalk_a"] == 3
    assert before["unsampled"]["sidewalk_a"]["tie"] - after["unsampled"]["sidewalk_a"]["tie"] == 1
    assert before["population"]["B"] == after["population"]["B"]  # flags for one city leave the other alone


def test_apply_author_excluded_joins_on_the_original_pool():
    p = _pool([{}, {}, {"expert1": "Agree"}, {}])
    flags = pd.DataFrame({"schema": ["sidewalk_a"] * 3, "label_id": [1, 2, 3], "author_excluded": ["f", "t", "f"]})
    out, missing = gr.apply_author_excluded(p, flags)
    assert missing == 1 and list(out.author_excluded) == [False, True, False, False]  # label 4: not in flags, kept
    assert list(out.index) == list(p.index) and list(out.label_uid) == list(p.label_uid)
    # eligibility follows the pinned pool's expert columns: a later prod vote by a rater is not in the flags
    assert list(gr.eligible(out).label_id) == [1, 4]
    # a pool that already carries the column has it replaced
    assert list(gr.apply_author_excluded(p.assign(author_excluded="t"), flags)[0].author_excluded) == \
        [False, True, False, False]
    with pytest.raises(ValueError, match="repeat"):
        gr.apply_author_excluded(p, pd.concat([flags, flags.head(1)]))


def test_score_table_flags_cells_under_min_cell():
    rows = ([{"label_type": "CurbRamp", "verdict": "incorrect"}] * 6 + [{"label_type": "CurbRamp", "verdict": "correct"}] * 2
            + [{"label_type": "Obstacle", "verdict": "correct"}]
            + [{"stratum": "light_voted", "label_type": "Obstacle", "verdict": "incorrect"}] * 2)
    m = _rated(rows)
    pop = {"lone": {"CurbRamp": 100, "Obstacle": 900, "Crosswalk": 50}, "light_voted": {"Obstacle": 10}}
    t = gr.score_table(m, pop).set_index(["stratum", "label_type"])
    assert t.loc[("lone", "all"), "w_rate"] == 0.75  # CurbRamp only: Obstacle (n=1) and Crosswalk (n=0) carry no weight
    assert t.loc[("lone", "all"), "w_excluded"] == "Obstacle(n=1), Crosswalk(n=0)"
    assert np.isnan(t.loc[("light_voted", "all"), "w_rate"])
    assert t.loc[("light_voted", "all"), "w_excluded"] == "Obstacle(n=2)"
    loose = gr.score_table(m, pop, min_cell=1).set_index(["stratum", "label_type"])
    assert loose.loc[("lone", "all"), "w_rate"] == pytest.approx((100 * 0.75 + 900 * 0.0) / 1000)
    md = gr.score_markdown(gr.score_table(m, pop))
    assert "| light_voted | all |" in md and "unestimable | Obstacle(n=2) |" in md
    assert "Obstacle(n=1), Crosswalk(n=0)" in md


def test_thin_stratum_covers_exactly_the_unsampled_disagrees():
    p = _pool(_city("sidewalk_a", n_per_stratum=3) + _thin_rows("sidewalk_a"))
    thin = gr.assign_stratum(p, thin=True)
    assert int((thin == "thin_voted").sum()) == sum(gr.unsampled_disagrees(p).values()) == 60
    plain = gr.assign_stratum(p)
    assert (thin[plain != ""] == plain[plain != ""]).all()  # every other row keeps its stratum
    assert set(thin[plain == ""]) == {"thin_voted"}
    assert gr.load_design(THIN_DESIGN)["shares"]["thin_voted"] == 0.2
    prep = gr.prepare(p, "sidewalk_a", THIN_DESIGN)
    assert (prep.cell == "thin_voted").sum() == 60
    q = gr.quotas(THIN_DESIGN)
    assert q[("thin_voted", "CurbRamp")] == 2 and sum(q[(c, "CurbRamp")] for c in
                                                       ("margin", "lone", "light_voted", "thin_voted")) == 10
    c = gr.candidates(prep, "seed", THIN_DESIGN, oversample=1)
    assert (c[(c.cell == "thin_voted") & (c.label_type == "CurbRamp")].shape[0]) == 2
    r = gr.recount_population(p, THIN_DESIGN)
    assert sum(r["population"]["A"]["thin_voted"].values()) == 60
    with pytest.raises(ValueError, match="shares"):
        gr.load_design({**DESIGN, "shares": {"margin": 0.5, "lone": 0.5, "other": 0.0}})


def test_round2_design_output_unchanged_with_thin_support():
    p = _pool(_city("sidewalk_a") + _thin_rows("sidewalk_a"))
    d2 = gr.load_design(gr.DEFAULT_DESIGN)
    assert not gr.uses_thin(d2) and not gr.uses_thin(DESIGN)
    a, b = gr.prepare(p, "sidewalk_a"), gr.prepare(p, "sidewalk_a", DESIGN)
    pd.testing.assert_frame_equal(a, b)
    assert "thin_voted" not in set(a.stratum) and set(a.cell) == set(gr.CELLS)
    assert set(gr.recount_population(p, DESIGN)["population"]["A"]) == set(gr.DISAGREE_STRATA)
    assert list(gr.candidates(a, "s", DESIGN).label_uid) == list(gr.candidates(b, "s", DESIGN).label_uid)


def _recount_files(tmp_path):
    p = _city("sidewalk_a", n_per_stratum=4) + _city("sidewalk_b", n_per_stratum=4) + _thin_rows("sidewalk_a")
    pool = _pool(p)
    pp, fp, dp = tmp_path / "pool.csv", tmp_path / "flags.csv", tmp_path / "design.json"
    pool.drop(columns=["label_uid", "h_dec", "crowd"]).to_csv(pp, index=False)
    a = pool[pool.schema == "sidewalk_a"]
    pd.DataFrame({"schema": a.schema, "label_id": a.label_id,
                  "author_excluded": np.where(a.label_id % 10 == 0, "t", "f")}).to_csv(fp, index=False)
    dp.write_text(json.dumps(DESIGN))
    return pool, pp, fp, dp


def test_cli_recount_writes_aggregates_only(tmp_path, capsys):
    pool, pp, fp, dp = _recount_files(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    a = pool[pool.schema == "sidewalk_a"]
    pd.DataFrame({"label_id": a.label_id[a.label_id % 10 != 0]}).to_csv(raw / "city-a.csv", index=False)
    conf = tmp_path / "cityparams.conf"
    conf.write_text('city-params {\n  db-schema {\n    city-a = "sidewalk_a"\n    city-b = "sidewalk_b"\n  }\n'
                    '  landing-page-url {\n    prod {\n      city-a = "https://a.example.edu"\n    }\n  }\n}\n')
    out, md = tmp_path / "recount.json", tmp_path / "recount.md"
    gr.main(["recount", "--pool", str(pp), "--author-excluded", str(fp), "--design", str(dp), "--out", str(out),
             "--markdown", str(md), "--rawlabels-proxy", str(raw), "--cityparams", str(conf)])
    printed = capsys.readouterr().out
    r = json.load(open(out))
    n_excl = int((a.label_id % 10 == 0).sum())
    assert r["excluded_labels"] == {"sidewalk_a": n_excl, "sidewalk_b": 0}
    dis_excl = int(((a.label_id % 10 == 0) & (a.ai_vote == "Disagree")).sum())
    assert r["excluded_eligible_disagrees"]["sidewalk_a"] == dis_excl
    assert r["pool_labels_missing_from_flags"] == int((pool.schema == "sidewalk_b").sum())
    prox = r["proxy_absent_from_export"]
    assert prox["sidewalk_a"] == {"ai_disagrees_absent": dis_excl, "of_which_flagged_excluded": dis_excl,
                                  "flagged_excluded_present_in_export": 0}
    assert prox["sidewalk_b"]["ai_disagrees_absent"] == 0  # no cache file for city-b: not counted as absent
    assert set(r) >= {"pool_sha256", "author_excluded_sha256", "design_sha256", "before", "after"}
    text = out.read_text()
    assert "sidewalk_a:" not in text and "label_id" not in text  # counts only, no label-level rows
    assert md.read_text().strip() == printed.split("\n-> ")[0].strip()
    assert "| sidewalk_a |" in printed and "before -> after" in printed


def test_cli_score_recounts_population_from_pool_and_flags(tmp_path, capsys):
    pool, pp, fp, dp = _recount_files(tmp_path)
    lone = pool[(pool.schema == "sidewalk_a") & (gr.assign_stratum(pool) == "lone") & (pool.label_id % 10 != 0)]
    a = lone.groupby("label_type").head(1)  # one rated item per type, none of them flagged excluded
    key = pd.DataFrame({"packet_id": [f"r{i}" for i in range(len(a))], "label_uid": a.label_uid.values,
                        "label_type": a.label_type.values, "stratum": "lone", "ai_vote": "Disagree",
                        "margin_class": "", "rater": "A", "shared": False})
    kp, sp = tmp_path / "key.csv", tmp_path / "sheet.csv"
    key.to_csv(kp, index=False)
    # The expert confirms the AI on Obstacle and Crosswalk only: the two types the flags thin out of the lone stratum.
    confirm = {"Obstacle", "Crosswalk"}
    pd.DataFrame({"packet_id": key.packet_id, "label_type": key.label_type,
                  "verdict": ["incorrect" if t in confirm else "correct" for t in key.label_type],
                  "notes": ""}).to_csv(sp, index=False)
    base = ["score", "--key", str(kp), "--rater", "A", "--sheet", str(sp), "--allow-unpinned"]

    def lone_row(extra):
        gr.main(base + extra)
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if ln.startswith("| lone | all |")]
        assert len(rows) == 1
        return out, rows[0]

    def pct(pop):
        return f"{100 * sum(pop[t] for t in confirm) / sum(pop.values()):.1f}%"

    before = gr.recount_population(pool, DESIGN)["population"]["A"]["lone"]
    after = gr.recount_population(gr.apply_author_excluded(pool, pd.read_csv(fp))[0], DESIGN)["population"]["A"]
    assert pct(after["lone"]) != pct(before)  # the fixture's flags must move the weighted rate

    # Default minimum n: 1 decided item per type, so every lone cell is named and the row is unestimable.
    out, row = lone_row(["--pool", str(pp), "--author-excluded", str(fp), "--design", str(dp)])
    assert f"population lone: ? -> {sum(after['lone'].values()):,}" in out
    assert sum(after["lone"].values()) < sum(before.values())
    assert "| unestimable |" in row and "Obstacle(n=1)" in row and "Crosswalk(n=1)" in row
    assert "WARNING" not in out

    # With every cell kept, the weighted rate is the recount's share of the confirmed types, not the pool's.
    _, flagged = lone_row(["--pool", str(pp), "--author-excluded", str(fp), "--design", str(dp), "--min-cell", "1"])
    _, unflagged = lone_row(["--pool", str(pp), "--design", str(dp), "--min-cell", "1"])
    _, no_pool = lone_row(["--min-cell", "1"])
    assert f"| {pct(after['lone'])} [" in flagged
    assert f"| {pct(before)} [" in unflagged
    assert flagged != unflagged and flagged != no_pool

    gr.main(base)
    assert "WARNING" not in capsys.readouterr().out  # no manifest population: nothing to warn about


# ---------------------------------------------------------------------------------------------------------------------
# #106: the three never-sampled AI-Disagree groups as strata of their own, check-design, slices in score
# ---------------------------------------------------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_SHARES = {"margin": 0, "lone": 0, "light_voted": 0, "tie": 0.45, "one_decisive": 0.33, "unsure_only": 0.22}
SPLIT_DESIGN = {**DESIGN, "shares": SPLIT_SHARES}
DRAFT = os.path.join(ROOT, "analysis", "gt_rounds", "unsampled_draft.json")
DESIGN_JSON = os.path.join(ROOT, "docs", "ai-disagree-relabel-2026-10", "unsampled_strata_design.json")
FLAGS = os.path.expanduser("~/Data/sidewalk-validator-ai-relabel-2026-10/round2/author_excluded.csv")
CONF = ('city-params {\n  db-schema {\n    city-a = "sidewalk_a"\n    city-b = "sidewalk_b"\n  }\n'
        '  landing-page-url {\n    prod {\n      city-a = "https://a.example.edu"\n      city-b = "https://b.example.edu"\n'
        '    }\n  }\n}\n')
VOTES = {"one_decisive": {"h_agree": 1}, "unsure_only": {"h_unsure": 2}, "tie": {"h_agree": 1, "h_dis": 1}}


def _counted(schema, counts):
    """AI Disagree rows of one city from ``{(group, type): n}``, ``group`` one of :data:`VOTES`' keys."""
    return [{"schema": schema, "label_type": t, "ai_vote": "Disagree", "severity": [1, 3, np.nan][i % 3],
             **VOTES[g]} for (g, t), n in counts.items() for i in range(n)]


def _keys(tree):
    """Every dict key in a nested JSON value."""
    if isinstance(tree, dict):
        return set(tree) | {k for v in tree.values() for k in _keys(v)}
    if isinstance(tree, list):
        return {k for v in tree for k in _keys(v)}
    return set()


def test_split_strata_partition_the_unsampled_disagrees():
    p = _pool(_city("sidewalk_a", n_per_stratum=3) + _thin_rows("sidewalk_a"))
    split = gr.assign_stratum(p, thin="split")
    u = gr.unsampled_disagrees(p)
    assert u == {"one_decisive": 20, "unsure_only": 20, "tie": 20}
    for g, n in u.items():
        assert int((split == g).sum()) == n
    plain = gr.assign_stratum(p)
    assert (split[plain != ""] == plain[plain != ""]).all()  # every other row keeps its stratum
    assert set(split[plain == ""]) == set(gr.UNSAMPLED_STRATA)
    assert (gr.unsampled_group(p) == split.where(plain == "", "")).all()
    assert gr.assign_stratum(p, thin="lumped").equals(gr.assign_stratum(p, thin=True))
    for bad in ("Split", 1, "thin_voted"):
        with pytest.raises(ValueError, match="thin"):
            gr.assign_stratum(p, thin=bad)


def test_unsampled_group_edge_cases():
    d = pd.DataFrame({"ai_vote": ["Disagree"] * 5 + ["Agree"], "h_agree": [1, 2, 1, 0, 0, 1],
                      "h_dis": [0, 2, 3, 0, 1, 1], "h_unsure": [2, 1, 0, 3, 0, 0]})
    assert list(gr.unsampled_group(d)) == ["one_decisive", "tie", "", "unsure_only", "one_decisive", ""]
    assert gr.assign_stratum(d)[2] == "margin_incorrect"  # 1-3 is a training row, not a tie
    with pytest.raises(ValueError, match="no unsampled group"):
        gr.unsampled_group(d.assign(h_unsure=[2, 1, 0, np.nan, 0, 0], h_dis=[0, 2, 3, np.nan, 1, 1]))


def test_load_design_accepts_split_shares_and_rejects_mixes():
    d = gr.load_design(SPLIT_DESIGN)
    assert gr.thin_mode(d) == "split" and gr.uses_thin(d)
    assert gr.thin_mode(THIN_DESIGN) == "lumped" and gr.thin_mode(DESIGN) is False and gr.thin_mode(None) is False
    with pytest.raises(ValueError, match="thin_voted"):
        gr.load_design({**DESIGN, "shares": {"margin": 0.4, "lone": 0.2, "light_voted": 0.2, "thin_voted": 0.1,
                                             "tie": 0.1}})
    with pytest.raises(ValueError, match="all three"):
        gr.load_design({**DESIGN, "shares": {"margin": 0.4, "lone": 0.2, "light_voted": 0.2, "tie": 0.1,
                                             "one_decisive": 0.1}})
    with pytest.raises(ValueError, match="shares"):
        gr.load_design({**DESIGN, "shares": {"tie": 0.5, "one_decisive": 0.3, "unsure_only": 0.2}})
    assert gr.load_design({**SPLIT_DESIGN, "draft": True})["draft"] is True
    with pytest.raises(ValueError, match="draft"):
        gr.load_design({**SPLIT_DESIGN, "draft": "yes"})
    q = gr.quotas(d)
    assert [q[(c, "CurbRamp")] for c in ("tie", "one_decisive", "unsure_only", "margin", "lone", "decoy")] == \
        [5, 3, 2, 0, 0, 4]


def test_split_design_prepares_and_recounts_the_three_strata():
    p = _pool(_city("sidewalk_a", n_per_stratum=3) + _thin_rows("sidewalk_a"))
    prep = gr.prepare(p, "sidewalk_a", SPLIT_DESIGN)
    assert {c: int((prep.cell == c).sum()) for c in gr.UNSAMPLED_STRATA} == {c: 20 for c in gr.UNSAMPLED_STRATA}
    pop = gr.recount_population(p, SPLIT_DESIGN)["population"]["A"]
    assert set(pop) == set(gr.DISAGREE_STRATA_SPLIT) and sum(pop["tie"].values()) == 20
    c = gr.candidates(prep, "seed", SPLIT_DESIGN, oversample=1)
    assert set(c.cell) == set(gr.UNSAMPLED_STRATA) | {"decoy"}  # zero quotas pick nothing
    # a lumped thin_voted design is unchanged: same strata, same candidates as before the split existed
    lumped = gr.prepare(p, "sidewalk_a", THIN_DESIGN)
    assert set(lumped.cell) == set(gr.CELLS) | {"thin_voted"} and (lumped.cell == "thin_voted").sum() == 60


def test_gold_table_by_stratum_keeps_thin_and_split_strata():
    key, sheets = _adj_case()
    key.loc[[4, 5], "stratum"] = ["tie", "thin_voted"]
    cons = pd.DataFrame({"packet_id": ["r2", "r3", "r4"], "consensus": ["correct", "incorrect", "incorrect"]})
    s = gr.gold_table(gr.adjudicated(key, sheets, cons), by="stratum").set_index("stratum")
    assert s.loc["tie", "shared_items"] == 1 and s.loc["tie", "gold"] == 1
    assert s.loc["thin_voted", "shared_items"] == 1 and s.loc["thin_voted", "no_gold"] == 1
    assert s.loc["all", "shared_items"] == sum(s.loc[x, "shared_items"] for x in s.index if x != "all")


def test_score_table_weights_a_split_stratum():
    m = _rated([{"stratum": "tie", "verdict": "incorrect"}] * 6 + [{"stratum": "tie", "label_type": "Obstacle",
                                                                   "verdict": "correct"}])
    t = gr.score_table(m, {"tie": {"CurbRamp": 100, "Obstacle": 50}}).set_index(["stratum", "label_type"])
    assert t.loc[("tie", "all"), "n"] == 7 and t.loc[("tie", "all"), "w_rate"] == 1.0
    assert t.loc[("tie", "all"), "w_excluded"] == "Obstacle(n=1)"


def test_score_population_defaults_to_the_manifest_design(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps(DESIGN))
    d, src = gr.score_design(None, {"design_json": SPLIT_DESIGN, "design_sha256": "x"}, False)
    assert gr.thin_mode(d) == "split" and "manifest" in src
    d, src = gr.score_design(None, {}, False)
    assert d == gr.load_design(gr.DEFAULT_DESIGN) and src == gr.DEFAULT_DESIGN  # an older manifest: round 2
    with pytest.raises(SystemExit, match="design"):
        gr.score_design(str(other), {"design_json": SPLIT_DESIGN, "design_sha256": "x"}, False)
    d, _ = gr.score_design(str(other), {"design_json": SPLIT_DESIGN, "design_sha256": "x"}, True)
    assert d == DESIGN
    pinned = {"design_json": DESIGN, "design_sha256": rl.sha256_file(other)}
    assert gr.score_design(str(other), pinned, False)[0] == DESIGN
    # round 2's own manifest names round 2's design, so its score output does not move
    man = json.load(open(os.path.join(ROUND2, "key", "manifest.json")))
    assert gr.score_design(None, man, False)[0] == gr.load_design(gr.DEFAULT_DESIGN)


def _split_round(tmp_path, monkeypatch, design=None, author_excluded=True):
    """A built split round on disk (crop check, fetch and marking monkeypatched): two cities, Obstacle's
    unsure_only cell in city A one label short of its quota."""
    rows = []
    for schema in ("sidewalk_a", "sidewalk_b"):
        rows += _city(schema, n_per_stratum=6)
        rows += _counted(schema, {(g, t): (1 if (schema, g, t) == ("sidewalk_a", "unsure_only", "Obstacle") else 12)
                                  for g in VOTES for t in gr.LABEL_TYPES})
    pool = _pool(rows).drop(columns=["label_uid", "h_dec", "crowd"])
    if author_excluded:
        pool["author_excluded"] = "f"
    pp, dp, conf = tmp_path / "pool.csv", tmp_path / "design.json", tmp_path / "cityparams.conf"
    pool.to_csv(pp, index=False)
    dp.write_text(json.dumps(design or SPLIT_DESIGN))
    conf.write_text(CONF)
    monkeypatch.setattr(rl, "crop_status", lambda host, c: ({}, {u: (10, 20) for u in c.label_uid}))
    monkeypatch.setattr(rl, "fetch_crops", lambda host, key, dir_map: {u: b"png" for u in key.label_uid})
    monkeypatch.setattr(rl, "marked_image", lambda png, x, y: b"jpg")
    kdir = tmp_path / "key"
    args = ["build", "--design", str(dp), "--pool", str(pp), "--packet-dir", str(tmp_path / "packet"),
            "--key-dir", str(kdir), "--seed", "s", "--cityparams", str(conf)]
    return args, pp, dp, kdir


def test_build_samples_the_split_strata_end_to_end(tmp_path, monkeypatch, capsys):
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch)
    gr.main(args)
    out = capsys.readouterr().out
    assert "(sampled as one_decisive / unsure_only / tie)" in out
    key = pd.read_csv(kdir / "key.csv", dtype={"tags": str})
    assert set(key.stratum) == set(gr.UNSAMPLED_STRATA) | {"decoy_agree"}
    man = json.load(open(kdir / "manifest.json"))
    assert set(man["population"]["A"]) == set(gr.DISAGREE_STRATA_SPLIT)
    assert man["population"]["A"]["tie"]["CurbRamp"] == 12 and man["population"]["A"]["unsure_only"]["Obstacle"] == 1
    assert man["shortfall"] == {"A:unsure_only/Obstacle": 1}
    q = gr.quotas(SPLIT_DESIGN)
    assert len(key[key.rater == "A"]) == sum(q.values()) - 1 and len(key[key.rater == "B"]) == sum(q.values())
    assert man["counts"]["A/tie/CurbRamp"] == q[("tie", "CurbRamp")]
    assert man["one_decisive_direction"]["A"]["CurbRamp"] == {"agree": 12, "disagree": 0}
    assert not any(c.startswith(("A/lone", "A/margin", "A/light")) for c in man["counts"])


def test_build_refuses_a_draft_and_a_pool_without_author_excluded(tmp_path, monkeypatch):
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch, design={**SPLIT_DESIGN, "draft": True})

    def no_remote(*a, **k):
        raise AssertionError("reached the database host")
    monkeypatch.setattr(rl, "crop_status", no_remote)
    with pytest.raises(SystemExit, match="draft"):
        gr.main(args)
    (tmp_path / "b").mkdir()
    args, pp, dp, kdir = _split_round(tmp_path / "b", monkeypatch, author_excluded=False)
    monkeypatch.setattr(rl, "crop_status", no_remote)
    with pytest.raises(SystemExit, match="author_excluded"):
        gr.main(args)
    assert not (kdir / "key.csv").exists()


def test_a_design_without_lone_draws_only_voted_decoys():
    """#106 review R2: every tie / one_decisive / unsure_only item has a human vote, so a decoy without one would
    mark itself as an AI Agree; a design with a lone share keeps every decoy, as round 2 did."""
    rows = _city("sidewalk_a", n_per_stratum=3) + _thin_rows("sidewalk_a")
    rows += [{"schema": "sidewalk_a", "label_type": t, "ai_vote": "Agree"} for t in gr.LABEL_TYPES for _ in range(4)]
    p = _pool(rows)
    votes = lambda d: d.h_agree + d.h_dis + d.h_unsure  # noqa: E731
    split = gr.prepare(p, "sidewalk_a", SPLIT_DESIGN)
    dec = split[split.cell == "decoy"]
    assert len(dec) == 15 and (votes(dec) > 0).all()  # 3 voted decoy_agree per type; unvoted Agree / Unsure out
    assert (votes(split[split.cell.isin(gr.UNSAMPLED_STRATA)]) > 0).all()  # what the design can draw
    old = gr.prepare(p, "sidewalk_a", DESIGN)
    assert (old.cell == "decoy").sum() == 50 and (votes(old[old.cell == "decoy"]) == 0).sum() == 35
    lumped = gr.prepare(p, "sidewalk_a", THIN_DESIGN)  # lone share > 0: unchanged
    assert (lumped.cell == "decoy").sum() == 50


def test_build_writes_direction_from_eligible_rows_and_records_the_exclusion(tmp_path, monkeypatch, capsys):
    """The manifest's one_decisive_direction counts eligible rows only (#106 review R5), and the manifest records
    that excluded authors' labels are out, so ``score`` without ``--pool`` does not warn (R4)."""
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch)
    pool = pd.read_csv(pp)
    od = pool[(pool.schema == "sidewalk_a") & (pool.label_type == "CurbRamp") & (pool.h_agree == 1)
              & (pool.h_dis == 0) & (pool.h_unsure == 0) & (pool.ai_vote == "Disagree")].index
    assert len(od) == 12
    pool.loc[od[:3], "author_excluded"] = "t"
    pool.loc[od[3:5], "expert1"] = "Agree"
    pool.to_csv(pp, index=False)
    gr.main(args)
    capsys.readouterr()
    man = json.load(open(kdir / "manifest.json"))
    assert man["one_decisive_direction"]["A"]["CurbRamp"] == {"agree": 7, "disagree": 0}
    assert man["population"]["A"]["one_decisive"]["CurbRamp"] == 7
    assert man["author_excluded_applied"] is True


def test_build_replays_a_pool_without_author_excluded_on_purpose(tmp_path, monkeypatch, capsys):
    """#106 review R7: round 2's pinned pool predates the column; --no-author-excluded builds from it, the manifest
    says the exclusion was not applied, and ``score`` without ``--pool`` warns about it (R4)."""
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch, author_excluded=False)
    gr.main(args + ["--no-author-excluded"])
    assert "WARNING: --no-author-excluded" in capsys.readouterr().out
    man = json.load(open(kdir / "manifest.json"))
    assert man["author_excluded_applied"] is False
    key = pd.read_csv(kdir / "key.csv", dtype={"tags": str})
    sp = tmp_path / "votes_A.csv"
    _write_sheet(sp, key, "A", lambda r: "incorrect" if r.ai_vote == "Disagree" else "correct")
    gr.main(["score", "--key", str(kdir / "key.csv"), "--rater", "A", "--sheet", str(sp), "--allow-unpinned"])
    assert "WARNING: population weights from the manifest, which count excluded authors' labels" in \
        capsys.readouterr().out


def test_cli_check_design_applies_the_pools_own_author_excluded_column(tmp_path, capsys, monkeypatch):
    """#106 review R5: the path every build-time pool takes. ``before`` ignores the column, ``after`` applies it."""
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch)
    pool = pd.read_csv(pp)
    tie = pool[(pool.schema == "sidewalk_a") & (pool.label_type == "CurbRamp") & (pool.h_agree == 1)
               & (pool.h_dis == 1)].index
    pool.loc[tie[:4], "author_excluded"] = "t"
    pool.to_csv(pp, index=False)
    out = tmp_path / "check.json"
    gr.main(["check-design", "--design", str(dp), "--pool", str(pp), "--out", str(out)])
    assert "from the pool's own column" in capsys.readouterr().out
    r = json.load(open(out))
    assert r["author_excluded_sha256"] is None and r["after"] is not None
    b, a = r["before"]["raters"]["A"], r["after"]["raters"]["A"]
    assert b["cells"]["tie"]["CurbRamp"]["population"] == 12 and a["cells"]["tie"]["CurbRamp"]["population"] == 8
    assert r["before"]["eligible_disagrees"]["sidewalk_a"] - r["after"]["eligible_disagrees"]["sidewalk_a"] == 4
    gr.main(["check-design", "--design", str(dp), "--pool", str(pp), "--no-author-excluded", "--out", str(out)])
    assert json.load(open(out))["after"] is None


def _write_sheet(path, key, rater, verdict):
    items = gr.rater_items(key, rater)
    pd.DataFrame({"packet_id": items.packet_id, "label_type": items.label_type,
                  "verdict": [verdict(r) for r in items.itertuples()], "notes": ""}).to_csv(path, index=False)


def test_score_on_a_split_round_weights_by_the_manifest_design_and_prints_slices(tmp_path, monkeypatch, capsys):
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch)
    gr.main(args)
    capsys.readouterr()
    key = pd.read_csv(kdir / "key.csv", dtype={"tags": str})
    sp = tmp_path / "votes_A.csv"
    _write_sheet(sp, key, "A", lambda r: "incorrect" if r.ai_vote == "Disagree" else "correct")
    base = ["score", "--key", str(kdir / "key.csv"), "--rater", "A", "--sheet", str(sp), "--allow-unpinned"]
    gr.main(base + ["--pool", str(pp)])  # no --design: the manifest's (split) design, not round 2's
    out = capsys.readouterr().out
    assert "population tie: 60 -> 60" in out and "population unsure_only: 49 -> 49" in out
    # the direction table carries each direction's population and share (12 single-Agree one_decisive per type)
    assert "| CurbRamp | Agree |" in out and "| 12 | 100% |" in out.split("| CurbRamp | Agree |")[1].splitlines()[0]
    assert "| all | Agree |" in out and "| 60 | 100% |" in out.split("| all | Agree |")[1].splitlines()[0]
    # the population of the sampled-strata slice counts the three sampled strata only (#106 review R3): CurbRamp
    # with no severity is 4 rows in each of tie / one_decisive / unsure_only, not the 4 lone / light / margin rows
    sec = out.split("### every sampled Disagree stratum, type x severity")[1].split("###")[0]
    row = [ln for ln in sec.splitlines() if ln.startswith("| CurbRamp | (none) |")][0]
    assert row.rstrip().endswith("| 12 |"), row
    tie = [ln for ln in out.splitlines() if ln.startswith("| tie | all |")][0]
    assert "100.0% [" in tie.split("|")[6]  # the population-weighted column is filled
    assert "## Slices" in out and "| CurbRamp | 1 |" in out and "(none)" in out
    assert "not population estimates" in out
    assert "single vote's direction" in out
    gr.main(base + ["--no-slices"])
    out = capsys.readouterr().out
    assert "## Slices" not in out and "single vote's direction" in out
    # without --pool: the manifest's population already leaves excluded authors out, so no warning (R4), and the
    # direction table takes the manifest's population
    assert "WARNING: population weights from the manifest" not in out
    assert "| 12 | 100% |" in out.split("| CurbRamp | Agree |")[1].splitlines()[0]
    rl.pin(str(kdir / "manifest.json"), "votes_A", str(sp), len(gr.rater_items(key, "A")))
    other = tmp_path / "round2_like.json"
    other.write_text(json.dumps(DESIGN))
    with pytest.raises(SystemExit, match="design"):
        gr.main(base[:-1] + ["--pool", str(pp), "--design", str(other)])


def test_score_slices_stay_off_for_a_key_without_split_strata(tmp_path, capsys):
    base, mp, pp, up = _cli_round(tmp_path)
    score = ["score", "--key", base[1], "--rater", "B", "--sheet", base[4].split("=", 1)[1], "--unmasked", str(up)]
    gr.main(score)
    out = capsys.readouterr().out
    assert "## Slices" not in out and "severity" not in out and "direction" not in out
    key = pd.read_csv(base[1])
    key.assign(severity=3, tags="narrow", crowd=np.nan).to_csv(base[1], index=False)  # a real key has these columns
    gr.main(score + ["--slices", "--allow-unpinned"])
    out = capsys.readouterr().out
    assert "## Slices" in out and "| CurbRamp | 3 |" in out and "| CurbRamp | narrow |" in out


def test_slice_rates_count_tags_twice_blank_small_cells_and_add_the_single_vote_baseline():
    base = {"stratum": "one_decisive", "severity": 1, "tags": "a;b", "crowd": "Agree", "h_agree": 1, "h_dis": 0,
            "verdict": "incorrect"}
    dis = {"crowd": "Disagree", "h_agree": 0, "h_dis": 1}
    m = _rated([base] * 3 + [{**base, **dis}] * 3 + [{**base, "verdict": "correct"}] + [{**base, "verdict": "cant_tell"}]
               + [{**base, "stratum": "tie", "crowd": np.nan, "h_dis": 1, "severity": 2, "tags": np.nan}]
               + [{**base, "stratum": "decoy_agree", "ai_vote": "Agree"}])
    train = pd.DataFrame({"label_type": "CurbRamp", "severity": [1] * 4, "tags": ["a"] * 4})
    t = gr.slice_rates(m, "tag", trains=[("round3_train", train)]).set_index("tag")
    for tag in ("a", "b"):  # a label with two tags counts once per tag
        assert (t.loc[tag, "items"], t.loc[tag, "n"], t.loc[tag, "k"], t.loc[tag, "cant_tell"]) == (8, 7, 6, 1)
        assert t.loc[tag, "rate"] == pytest.approx(6 / 7)
        # the single vote matches the expert 4 times of 7 (3 Disagree + 1 Agree on a correct label), mismatches 3
        assert (t.loc[tag, "base_n"], t.loc[tag, "base_k"]) == (7, 4)
    assert t.loc["a", "round3_train"] == 4 and t.loc["b", "round3_train"] == 0
    assert t.loc["(none)", "n"] == 1 and np.isnan(t.loc["(none)", "rate"]) and t.loc["(none)", "base_n"] == 0
    s = gr.slice_rates(m, "severity", strata=["tie"])
    assert list(s.severity) == ["2"] and s.n.iloc[0] == 1 and np.isnan(s.rate.iloc[0])
    md = gr.slices_markdown(t.reset_index(), "tag", ("round3_train",))
    assert "| CurbRamp | a | 8 | 85.7% [" in md and "round3_train" in md and "r0" not in md


def test_direction_rates_split_one_decisive_by_the_single_vote():
    m = _rated([{"stratum": "one_decisive", "h_agree": 1, "h_dis": 0, "verdict": "incorrect"}] * 5
               + [{"stratum": "one_decisive", "h_agree": 0, "h_dis": 1, "verdict": "correct"}] * 2
               + [{"stratum": "tie", "h_agree": 1, "h_dis": 1, "verdict": "incorrect"}])
    t = gr.direction_rates(m, {"CurbRamp": {"agree": 30, "disagree": 10}}).set_index(["label_type", "direction"])
    assert t.loc[("all", "Agree"), "n"] == 5 and t.loc[("all", "Agree"), "rate"] == 1.0
    assert t.loc[("all", "Disagree"), "n"] == 2 and np.isnan(t.loc[("all", "Disagree"), "rate"])
    assert t.loc[("CurbRamp", "Agree"), "pop_share"] == 0.75 and t.loc[("all", "Disagree"), "population"] == 10


def test_design_check_statuses_precedence_and_half_width():
    design = {**SPLIT_DESIGN, "per_type": {t: 20 for t in gr.LABEL_TYPES}}  # tie 9, one_decisive 7, unsure_only 4
    q = gr.quotas(design)
    assert (q[("tie", "CurbRamp")], q[("one_decisive", "CurbRamp")], q[("unsure_only", "CurbRamp")]) == (9, 7, 4)
    a = _counted("sidewalk_a", {("tie", "CurbRamp"): 30, ("tie", "Obstacle"): 6, ("one_decisive", "CurbRamp"): 30,
                                ("one_decisive", "Obstacle"): 10, ("unsure_only", "CurbRamp"): 30,
                                ("unsure_only", "Crosswalk"): 3})
    b = _counted("sidewalk_b", {("tie", "CurbRamp"): 2})
    p = _pool(a + b + _city("sidewalk_a", n_per_stratum=1))
    r = gr.design_check(p, design)
    cells = r["raters"]["A"]["cells"]
    assert cells["tie"]["CurbRamp"] == {"population": 30, "quota": 9, "available": 9, "expected_decided": 6,
                                        "need": 7, "status": "ok", "shortfall": 0}
    assert cells["tie"]["Obstacle"]["status"] == "short" and cells["tie"]["Obstacle"]["shortfall"] == 3
    assert cells["tie"]["Obstacle"]["expected_decided"] == 4
    assert cells["unsure_only"]["CurbRamp"]["status"] == "under_min"
    assert cells["unsure_only"]["Crosswalk"]["status"] == "no_weight"  # population 3 < 5 beats short (3 of 4)
    assert cells["tie"]["NoCurbRamp"]["status"] == "no_weight" and "margin" not in cells  # zero quota: not listed
    st = r["raters"]["A"]["strata"]
    assert st["one_decisive"]["half_width"] == pytest.approx(1.96 * np.sqrt((0.75 ** 2 + 0.25 ** 2) * 0.25 / 5))
    assert st["one_decisive"]["kish_n"] == pytest.approx(8.0)
    assert st["tie"]["weighted_types"] == ["CurbRamp"]  # Obstacle expects 4 decided: no weight
    assert st["tie"]["half_width"] == pytest.approx(1.96 * np.sqrt(0.25 / 6))
    assert r["raters"]["A"]["one_decisive_direction"]["CurbRamp"] == {"agree": 30, "disagree": 0}
    own_a = r["raters"]["A"]["own_items"]
    # one decoy per type: _city's decoy_unsure rows have no human vote, and a design without lone drops them
    assert own_a == sum(c["available"] for t in cells.values() for c in t.values()) + 5
    assert r["raters"]["A"]["decoys"]["CurbRamp"]["population"] == 1
    assert r["raters"]["A"]["rated_items"] == own_a + min(6, r["raters"]["B"]["own_items"])
    rc = gr.recount_population(p, design)
    assert r["eligible_disagrees"] == rc["eligible_disagrees"] and r["unsampled"] == rc["unsampled"]
    sev = {(x["label_type"], x["severity"]): x for x in r["raters"]["A"]["slices"]["severity"]}
    assert sev[("CurbRamp", "1")]["tie"] == 10 and sev[("CurbRamp", "(none)")]["all"] == 30
    assert json.loads(json.dumps(r)) == r  # JSON-native: committed verbatim


def test_cli_check_design_needs_exclusion_info_and_writes_aggregates_only(tmp_path, capsys, monkeypatch):
    args, pp, dp, kdir = _split_round(tmp_path, monkeypatch, author_excluded=False)
    run = ["check-design", "--design", str(dp), "--pool", str(pp)]
    with pytest.raises(SystemExit, match="author_excluded"):
        gr.main(run)
    out = tmp_path / "check.json"
    gr.main(run + ["--no-author-excluded", "--out", str(out)])
    r = json.load(open(out))
    assert r["after"] is None and r["author_excluded_sha256"] is None
    pool = pd.read_csv(pp)
    a = pool[(pool.schema == "sidewalk_a") & (pool.h_agree == 1) & (pool.h_dis == 1)]
    flags = tmp_path / "flags.csv"
    pd.DataFrame({"schema": pool.schema, "label_id": pool.label_id,
                  "author_excluded": np.where(pool.label_id.isin(a.label_id.head(4)) & (pool.schema == "sidewalk_a"),
                                              "t", "f")}).to_csv(flags, index=False)
    md = tmp_path / "check.md"
    gr.main(run + ["--author-excluded", str(flags), "--out", str(out), "--markdown", str(md)])
    printed = capsys.readouterr().out
    r = json.load(open(out))
    assert r["pool_sha256"] == rl.sha256_file(pp) and r["design_sha256"] == rl.sha256_file(dp)
    assert r["author_excluded_sha256"] == rl.sha256_file(flags) and r["cant_tell"] == 0.25 and r["min_cell"] == 5
    tb = r["before"]["raters"]["A"]["cells"]["tie"]["CurbRamp"]["population"]
    assert tb - r["after"]["raters"]["A"]["cells"]["tie"]["CurbRamp"]["population"] == 4
    assert not any(k in ("label_id", "label_uid") or k.endswith("_id") for k in _keys(r))
    assert "sidewalk_a:" not in out.read_text()
    assert "Crop availability is not checked" in printed and "| tie | CurbRamp |" in printed and md.read_text().strip() in printed


def test_draft_design_in_repo_is_a_draft_of_250_per_city():
    d = gr.load_design(DRAFT)
    assert d["draft"] is True and gr.thin_mode(d) == "split"
    assert not any(k.lower().endswith("_id") for k in d)  # no rater ids in the design
    q = gr.quotas(d)
    assert sum(q.values()) == 250
    dis = {k: n for k, n in q.items() if k[0] != "decoy" and n}
    assert {c for c, _ in dis} == set(gr.UNSAMPLED_STRATA) and min(dis.values()) >= 8
    assert [q[(c, "CurbRamp")] for c in ("tie", "one_decisive", "unsure_only")] == [16, 12, 8]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROUND2, "pool.csv.gz")), reason="committed round-2 pool absent")
def test_committed_design_check_reproduces_from_the_committed_pool():
    r = json.load(open(DESIGN_JSON))
    pool = os.path.join(ROUND2, "pool.csv.gz")
    assert r["pool_sha256"] == rl.sha256_file(pool)
    sums = dict(reversed(ln.split()) for ln in open(os.path.join(ROUND2, "SHA256SUMS")) if ln.strip())
    assert r["pool_sha256"] == sums["pool.csv.gz"]
    assert r["design_sha256"] == rl.sha256_file(DRAFT)
    design = gr.load_design(DRAFT)
    before = gr.design_check(rl.load_pool(pool).drop(columns=["author_excluded"], errors="ignore"), design,
                             r["min_cell"], r["cant_tell"])
    assert json.loads(json.dumps(before)) == r["before"]
    rc = json.load(open(os.path.join(ROOT, "docs", "ai-disagree-relabel-2026-10", "round2_recount.json")))
    assert r["after"]["unsampled"] == rc["after"]["unsampled"]
    assert r["after"]["eligible_disagrees"] == rc["after"]["eligible_disagrees"]
    assert r["before"]["unsampled"] == rc["before"]["unsampled"]
    assert r["author_excluded_sha256"] == rc["author_excluded_sha256"]
    for rater in design["raters"]:
        for g in gr.UNSAMPLED_STRATA:  # the after cells partition the recount's unsampled counts
            schema = design["raters"][rater]
            assert r["after"]["raters"][rater]["strata"][g]["population"] == rc["after"]["unsampled"][schema][g]
    assert not any(k in ("label_id", "label_uid") or k.endswith("_id") for k in _keys(r))


@pytest.mark.skipif(not (os.path.exists(FLAGS) and os.path.exists(os.path.join(ROUND2, "pool.csv.gz"))),
                    reason="author_excluded flags are not committed (label ids); local only")
def test_committed_design_check_after_reproduces_with_the_local_flags():
    r = json.load(open(DESIGN_JSON))
    assert rl.sha256_file(FLAGS) == r["author_excluded_sha256"]
    pool, missing = gr.apply_author_excluded(rl.load_pool(os.path.join(ROUND2, "pool.csv.gz")), pd.read_csv(FLAGS))
    after = gr.design_check(pool, gr.load_design(DRAFT), r["min_cell"], r["cant_tell"])
    assert json.loads(json.dumps(after)) == r["after"] and missing == r["pool_labels_missing_from_flags"]


def test_design_doc_quotes_the_committed_check():
    """The page's population rows, precision rows, item counts and pins are the committed JSON's."""
    r = json.load(open(DESIGN_JSON))
    doc = open(os.path.join(ROOT, "docs", "ai-disagree-relabel-2026-10", "unsampled_strata_design.md")).read()
    assert f"`{rl.sha256_file(DESIGN_JSON)}`" in doc and f"`{r['design_sha256']}`" in doc
    assert f"`{r['pool_sha256']}`" in doc and f"`{r['author_excluded_sha256']}`" in doc
    flat = " ".join(doc.split())
    for rater, city in (("jon", "Seattle"), ("mikey", "Chicago")):
        x = r["after"]["raters"][rater]
        for g in gr.UNSAMPLED_STRATA:
            row = " | ".join(f"{x['cells'][g][t]['population']:,}" for t in gr.LABEL_TYPES)
            assert f"| {g} | {row} | {x['strata'][g]['population']:,} |" in doc, (city, g)
            s = x["strata"][g]
            # half-width and Kish n bound to the stratum's own row (#106 review R9)
            row = re.compile(rf"\| {g} \| {s['expected_decided']} \| ±{100 * s['half_width']:.1f}( points)? \| "
                             rf"{round(s['kish_n'])} \| ")
            assert row.search(doc), (city, g)
            for t, c in x["cells"][g].items():  # every cell that is not ok is named in the cells table
                assert c["status"] == "ok" or f"**{c['status']}**" in doc, (city, g, t)
        assert f"{x['own_items']} and" in flat or f"and {x['own_items']} own-city" in flat, city
        assert f"({x['rated_items']} and" in flat or f"and {x['rated_items']} rated" in flat, city
