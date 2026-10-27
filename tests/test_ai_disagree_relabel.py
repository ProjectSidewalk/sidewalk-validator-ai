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
            "pano_id": "p", "heading": 0.0, "pitch": 0.0}
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


def test_candidates_exclude_rater_votes_and_rank_agree_before_unsure():
    rows = ([{"ai_vote": "Agree"} for _ in range(3)] + [{"ai_vote": "Unsure"} for _ in range(3)]
            + [{"ai_vote": "Agree", "expert1": "Agree"}])
    c = rl.candidates(_pool(rows), seed="s", per_cell=2, oversample=2)
    assert len(c) == 4 and (c.cell == "decoy").all()
    assert list(c.group) == ["decoy_agree"] * 3 + ["decoy_unsure"]
    assert c.expert1.isna().all()


def test_candidates_are_deterministic_in_the_seed():
    rows = [{} for _ in range(50)]
    a = rl.candidates(_pool(rows), "s1", per_cell=5, oversample=1).label_uid.tolist()
    assert a == rl.candidates(_pool(rows), "s1", per_cell=5, oversample=1).label_uid.tolist()
    assert a != rl.candidates(_pool(rows), "s2", per_cell=5, oversample=1).label_uid.tolist()


def test_pick_skips_unavailable_and_reports_shortfall():
    c = rl.candidates(_pool([{} for _ in range(4)]), "s", per_cell=3, oversample=2)
    avail = set(c.label_uid[:2])
    picked, short = rl.pick(c, avail, per_cell=3)
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
    m = rl.schema_dir_map(["sidewalk_columbus", "sidewalk_new_taipei", "sidewalk_taipei", "sidewalk_x"],
                          ["columbus-oh", "new-taipei-tw", "taipei", "x-a", "x-b"])
    assert m == {"sidewalk_columbus": "columbus-oh", "sidewalk_new_taipei": "new-taipei-tw",
                 "sidewalk_taipei": "taipei"}
    assert rl.crop_path(m, "sidewalk_taipei", "Obstacle", 12) == "/www/sidewalk/prod/images/taipei/Obstacle/crop_12.png"
    assert rl.crop_path(m, "sidewalk_x", "Obstacle", 12) is None


def _key():
    return pd.DataFrame({
        "packet_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "group": ["lone_disagree", "lone_disagree", "voted_disagree", "voted_disagree", "decoy_agree", "decoy_unsure"],
        "label_type": ["CurbRamp"] * 6,
        "ai_vote": ["Disagree", "Disagree", "Disagree", "Disagree", "Agree", "Unsure"],
        "crowd": [None, None, "Disagree", "Agree", "Agree", None]})


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
    assert voted.crowd_rate == 0.5               # the crowd got r4 (Agree) right, r3 (Disagree) wrong
    assert t.loc[("decoy_agree", "all")].rate == 1.0
    assert t.loc[("decoy_unsure", "all")].k == 1
    assert ("lone_disagree", "Obstacle") not in t.index


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
