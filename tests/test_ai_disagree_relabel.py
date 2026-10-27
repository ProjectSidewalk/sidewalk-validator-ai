"""Tests for analysis/ai_disagree_relabel.py: sampling, masking, unmasking, scoring and the severity/tag breakdown
(the server calls are thin and run by hand)."""
import doctest
import io

import numpy as np
import pandas as pd
import pytest

from analysis import ai_disagree_relabel as rl


def _pool(rows):
    """A pool DataFrame as ``load_pool`` returns it, from dicts with defaults for the columns a test does not set."""
    base = {"schema": "sidewalk_x", "label_type": "CurbRamp", "severity": 1, "tags": np.nan, "ai_vote": "Disagree",
            "ai_accuracy": 0.95, "h_agree": 0, "h_dis": 0, "h_unsure": 0, "expert1": np.nan, "expert2": np.nan,
            "expert1_labeled": "f", "pano_id": "p", "heading": 0.0, "pitch": 0.0}
    df = pd.DataFrame([{**base, "label_id": i + 1, **r} for i, r in enumerate(rows)])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return rl.load_pool(buf)


def test_doctests():
    assert doctest.testmod(rl).failed == 0


def test_load_pool_keys_on_city_and_label_id():
    p = _pool([{"schema": "sidewalk_a", "label_id": 7}, {"schema": "sidewalk_b", "label_id": 7}])
    assert list(p.label_uid) == ["sidewalk_a:7", "sidewalk_b:7"]
    with pytest.raises(ValueError, match="not unique"):
        _pool([{"label_id": 7}, {"label_id": 7}])


def test_crowd_majority_and_ties():
    p = _pool([{"h_agree": 2, "h_dis": 1}, {"h_agree": 1, "h_dis": 1}, {"h_agree": 0, "h_dis": 3}])
    assert p.crowd[0] == "Agree" and pd.isna(p.crowd[1]) and p.crowd[2] == "Disagree"


def test_groups_need_two_decisive_votes_and_lone_means_no_vote_at_all():
    d = pd.DataFrame({"ai_vote": ["Disagree"] * 4, "h_agree": [1, 0, 0, 2], "h_dis": [0, 0, 0, 0],
                      "h_unsure": [0, 1, 0, 3]})
    assert list(rl.assign_group(d)) == [None, None, "lone_disagree", "voted_disagree"]


def test_candidates_exclude_labels_the_rater_voted_on_or_placed():
    rows = [{}, {"expert1": "Agree"}, {"expert1_labeled": "t"}, {"ai_vote": "Agree"},
            {"ai_vote": "Agree", "expert1_labeled": "t"}]
    c = rl.candidates(_pool(rows), seed="s", per_cell=2, oversample=2)
    assert sorted(c.label_id) == [1, 4]


@pytest.mark.parametrize("seed", [f"s{i}" for i in range(12)])
def test_decoys_rank_agree_before_unsure_within_a_city(seed):
    rows = [{}] + [{"ai_vote": "Unsure"} for _ in range(6)] + [{"ai_vote": "Agree"}]
    c = rl.candidates(_pool(rows), seed=seed, per_cell=1, oversample=2)
    dec = c[c.cell == "decoy"]
    assert list(dec.group) == ["decoy_agree", "decoy_unsure"]
    # pick must prefer the Agree even when the Unsure ranks first
    c.loc[c.group == "decoy_unsure", "rank"] = 0
    picked, _ = rl.pick(c, set(c.label_uid), seed, per_cell=1)
    assert list(picked[picked.cell == "decoy"].ai_vote) == ["Agree"]


def test_scarce_agree_decoy_gets_first_pass_over_all_slots():
    # Two Disagree items, one in each city; the only Agree decoy is in the city whose slot may come second.
    rows = ([{"schema": "sidewalk_a"}, {"schema": "sidewalk_b", "h_agree": 2}]
            + [{"schema": "sidewalk_a", "ai_vote": "Unsure"}] * 3 + [{"schema": "sidewalk_b", "ai_vote": "Agree"}])
    for seed in (f"s{i}" for i in range(12)):
        c = rl.candidates(_pool(rows), seed=seed, per_cell=1, oversample=3)
        picked, _ = rl.pick(c, set(c.label_uid), seed, per_cell=1)
        assert list(picked[picked.cell == "decoy"].ai_vote) == ["Agree"]


def test_decoys_stop_at_per_cell_with_more_slots_than_needed():
    rows = [{} for _ in range(4)] + [{"h_agree": 2} for _ in range(4)] + [{"ai_vote": "Agree"} for _ in range(10)]
    c = rl.candidates(_pool(rows), seed="s", per_cell=3, oversample=4)
    picked, short = rl.pick(c, set(c.label_uid), "s", per_cell=3)
    assert (picked.cell == "decoy").sum() == 3 and "decoy/CurbRamp" not in short


def test_decoy_slot_order_is_seeded_not_lone_first():
    # lone items all in city a, voted in city b; with one decoy, a seeded slot order must sometimes pick city b.
    rows = ([{"schema": "sidewalk_a"}] * 3 + [{"schema": "sidewalk_b", "h_agree": 2}] * 3
            + [{"schema": s, "ai_vote": "Agree"} for s in ("sidewalk_a", "sidewalk_b") for _ in range(3)])
    cities = set()
    for seed in (f"s{i}" for i in range(20)):
        c = rl.candidates(_pool(rows), seed=seed, per_cell=3, oversample=2)
        picked, _ = rl.pick(c, set(c.label_uid), seed, per_cell=1)
        cities |= set(picked[picked.cell == "decoy"].schema)
    assert cities == {"sidewalk_a", "sidewalk_b"}


def test_population_counts_only_eligible_labels():
    p = _pool([{}, {}, {"expert1": "Agree"}, {"expert1_labeled": "t"}, {"h_dis": 2}])
    pop = rl.population(p)
    assert pop["lone_disagree"]["CurbRamp"] == 2 and pop["voted_disagree"]["CurbRamp"] == 1


def test_decoys_follow_the_disagree_items_cities():
    rows = ([{"schema": "sidewalk_a"} for _ in range(3)] + [{"schema": "sidewalk_b", "h_agree": 2}]
            + [{"schema": s, "ai_vote": "Agree"} for s in ("sidewalk_a", "sidewalk_b", "sidewalk_c")
               for _ in range(5)])
    c = rl.candidates(_pool(rows), seed="s", per_cell=4, oversample=5)
    assert set(c[c.cell == "decoy"].schema) == {"sidewalk_a", "sidewalk_b"}   # no Disagree item in sidewalk_c
    picked, short = rl.pick(c, set(c.label_uid), "s", per_cell=4)
    dec = picked[picked.cell == "decoy"]
    assert len(dec) == 4 and dec.schema.value_counts().to_dict() == {"sidewalk_a": 3, "sidewalk_b": 1}
    assert dec.label_uid.is_unique


def test_candidates_are_deterministic_in_the_seed():
    rows = [{} for _ in range(50)]
    a = rl.candidates(_pool(rows), "s1", per_cell=5, oversample=1).label_uid.tolist()
    assert a == rl.candidates(_pool(rows), "s1", per_cell=5, oversample=1).label_uid.tolist()
    assert a != rl.candidates(_pool(rows), "s2", per_cell=5, oversample=1).label_uid.tolist()


def test_pick_skips_unavailable_and_reports_shortfall():
    c = rl.candidates(_pool([{} for _ in range(4)]), "s", per_cell=3, oversample=2)
    avail = set(c.label_uid[:2])
    picked, short = rl.pick(c, avail, "s", per_cell=3)
    assert set(picked.label_uid) == avail
    assert short["lone_disagree/CurbRamp"] == 1
    assert short["decoy/Obstacle"] == 3


def test_make_key_masks_with_unique_ids_and_is_seeded():
    picked = rl.candidates(_pool([{} for _ in range(30)]), "s", per_cell=30, oversample=1)
    k1, k2 = rl.make_key(picked, "secret"), rl.make_key(picked, "secret")
    assert k1.packet_id.is_unique and k1.packet_id.str.match(r"^r[0-9a-f]{6}$").all()
    assert k1.equals(k2)
    assert not k1.label_uid.equals(picked.sort_values("label_uid").label_uid.reset_index(drop=True))
    assert not k1.packet_id.equals(rl.make_key(picked, "other").packet_id)


def test_schema_dir_map_prefixes_and_ambiguity():
    m = rl.schema_dir_map(["sidewalk_foo", "sidewalk_new_bar", "sidewalk_bar", "sidewalk_x"],
                          ["foo-xx", "new-bar-yy", "bar", "x-a", "x-b"])
    assert m == {"sidewalk_foo": "foo-xx", "sidewalk_new_bar": "new-bar-yy", "sidewalk_bar": "bar"}
    assert rl.crop_path(m, "sidewalk_bar", "Obstacle", 12) == "/www/sidewalk/prod/images/bar/Obstacle/crop_12.png"
    assert rl.crop_path(m, "sidewalk_x", "Obstacle", 12) is None


def _key():
    return pd.DataFrame({
        "packet_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "group": ["lone_disagree", "lone_disagree", "voted_disagree", "voted_disagree", "decoy_agree", "decoy_unsure"],
        "label_type": ["CurbRamp"] * 6,
        "ai_vote": ["Disagree", "Disagree", "Disagree", "Disagree", "Agree", "Unsure"],
        "crowd": [None, None, "Agree", "Agree", "Agree", None]})


def test_unmask_requires_exact_coverage_and_known_verdicts():
    sheet = pd.DataFrame({"packet_id": ["r1", "r2", "r3", "r4", "r5", "r6"], "verdict": ["incorrect"] * 6})
    assert len(rl.unmask(_key(), sheet)) == 6
    with pytest.raises(ValueError, match="do not match"):
        rl.unmask(_key(), sheet.iloc[:5])
    with pytest.raises(ValueError, match="unexpected"):
        rl.unmask(_key(), sheet.assign(verdict=["maybe"] + ["incorrect"] * 5))


def test_score_table_precision_cant_tell_and_crowd():
    sheet = pd.DataFrame({"packet_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                          "verdict": ["incorrect", "cant_tell", "correct", "correct", "correct", "correct"]})
    t = rl.score_table(rl.unmask(_key(), sheet)).set_index(["group", "label_type"])
    lone = t.loc[("lone_disagree", "all")]
    assert (lone.n, lone.k, lone.cant_tell) == (1, 1, 1)
    voted = t.loc[("voted_disagree", "all")]
    assert (voted.n, voted.k) == (2, 0)          # the expert called both correct: both AI Disagrees wrong
    assert voted.crowd_rate == 1.0               # crowd said Agree on both and the expert called both correct
    assert t.loc[("decoy_agree", "all")].rate == 1.0
    assert t.loc[("decoy_unsure", "all")].k == 1
    assert ("lone_disagree", "Obstacle") not in t.index


def test_score_table_population_weighting():
    key = pd.DataFrame({"packet_id": [f"r{i}" for i in range(4)], "group": ["lone_disagree"] * 4,
                        "label_type": ["CurbRamp", "CurbRamp", "Obstacle", "Obstacle"], "ai_vote": ["Disagree"] * 4,
                        "crowd": [None] * 4})
    sheet = pd.DataFrame({"packet_id": key.packet_id, "verdict": ["incorrect", "incorrect", "correct", "correct"]})
    pop = {"lone_disagree": {"CurbRamp": 900, "Obstacle": 100}}
    row = rl.score_table(rl.unmask(key, sheet), pop).set_index(["group", "label_type"]).loc[("lone_disagree", "all")]
    assert row.rate == 0.5 and row.w_rate == pytest.approx(0.9)


def test_wilson_interval_is_95_percent():
    r = rl._rate(8, 10)
    assert (round(r["lo"], 4), round(r["hi"], 4)) == (0.4902, 0.9433)


def test_confirm_rows_need_two_decisive_votes_and_a_majority():
    p = _pool([{"h_dis": 1}, {"h_agree": 1, "h_dis": 1}, {"h_dis": 2}, {"h_agree": 2, "h_dis": 1}])
    c = rl.confirm_rows(p)
    assert list(c.label_id) == [3, 4] and list(c.confirmed) == [True, False]


def test_breakdown_keeps_missing_severity_and_lone_counts_respect_unsure():
    rows = ([{"severity": np.nan, "h_dis": 2}] * 2 + [{"severity": np.nan}] * 3
            + [{"severity": np.nan, "h_unsure": 1}])
    p = _pool(rows)
    b = rl.breakdown(p, "severity", min_n=1)
    assert list(b.severity) == ["(none)"] and b.Disagree_n.iloc[0] == 2
    assert rl.lone_counts(p, "severity").lone_disagree.tolist() == [3]


def test_breakdown_markdown_keeps_lone_only_cells():
    p = _pool([{"tags": "rare"}] * 3 + [{"tags": "common", "h_dis": 2}] * 3)
    md = rl.breakdown_markdown(p, min_n=3)
    assert "| CurbRamp | rare |" in md and "| CurbRamp | common |" in md


def test_breakdown_explodes_tags_and_blanks_small_cells():
    rows = ([{"ai_vote": "Disagree", "h_dis": 2, "tags": "steep;narrow"}] * 3
            + [{"ai_vote": "Disagree", "h_agree": 2, "tags": "steep"}] * 1
            + [{"ai_vote": "Agree", "h_agree": 2}] * 2
            + [{"ai_vote": "Disagree"}] * 5)            # lone: no human votes, not in the rates
    p = _pool(rows)
    b = rl.breakdown(p, "tag", min_n=2).set_index("tag")
    assert b.loc["steep", "Disagree_n"] == 4 and b.loc["steep", "Disagree_rate"] == 0.75
    assert b.loc["narrow", "Disagree_n"] == 3
    assert np.isnan(b.loc["(none)", "Disagree_rate"]) and b.loc["(none)", "Agree_rate"] == 1.0
    lone = rl.lone_counts(p, "severity")
    assert lone.lone_disagree.sum() == 5
    assert "| CurbRamp | steep |" in rl.breakdown_markdown(p, min_n=2)


def test_marked_image_downscales_to_jpeg():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), (128, 128, 128)).save(buf, "PNG")
    out = Image.open(io.BytesIO(rl.marked_image(buf.getvalue(), 0.5, 0.5, max_width=1000)))
    assert out.format == "JPEG" and out.size == (1000, 500)
    assert out.getpixel((500, 250)) != out.getpixel((500 + 35, 250))  # ring drawn at the marker, not the centre fill


def test_street_view_url():
    assert rl.street_view_url("abc", 10, -5.5).endswith("pano=abc&heading=10.00&pitch=-5.50")
