"""analysis/ai_labels_relabel.py: cells, the seeded pick, prior-vote parsing, the Gallery URL and the scoring join."""
import doctest
import json

import pandas as pd
import pytest

from analysis import ai_labels_relabel as rl


def vote(user, v, vt="Human"):
    return {"user_id": user, "validation": v, "validator_type": vt}


def test_doctests():
    failed, _ = doctest.testmod(rl)
    assert failed == 0


def test_pick_is_stable_under_pool_growth_and_takes_every_non_decoy():
    pool = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(40)],
                         "verdict": ["incorrect"] * 6 + ["correct"] * 34,
                         "true": ["correct"] * 3 + ["incorrect"] * 3 + ["incorrect"] * 4 + ["correct"] * 30})
    a = rl.pick(pool, "s", n_decoys=5)
    assert a.cell.value_counts().to_dict() == {"decoy": 5, "contested": 3, "confirmed": 3, "missed": 4}
    assert a.packet_id.tolist() == list(range(1, 16))
    # a new decoy candidate entering the pool changes the chosen decoys by at most one
    bigger = pd.concat([pool, pd.DataFrame({"label_uid": ["c:99"], "verdict": ["correct"], "true": ["correct"]})])
    b = rl.pick(bigger, "s", n_decoys=5)
    assert len(set(a[a.cell == "decoy"].label_uid) ^ set(b[b.cell == "decoy"].label_uid)) <= 2
    assert set(a[a.cell != "decoy"].label_uid) == set(b[b.cell != "decoy"].label_uid)


def test_prior_votes_takes_the_raters_last_human_vote_and_ignores_ai():
    v = json.dumps([vote("r", "Agree"), vote("r", "Disagree"), vote("ai", "Agree", "AI"), vote("x", "Unsure")])
    assert rl.prior_votes(v, "r") == ("Disagree", ["r", "r", "x"])
    assert rl.prior_votes("[]", "r") == ("", [])


def test_gallery_url_caps_at_500():
    assert rl.gallery_url("https://x", [3, 1]) == "https://x/gallery?labelIds=3,1"
    with pytest.raises(ValueError):
        rl.gallery_url("https://x", list(range(501)))


def test_mask_console_js_contains_every_selector():
    js = rl.mask_console_js()
    for sel in (".card-validation-info", ".label-detail__vote-count", "#ps-tooltip", ".ai-icon"):
        assert sel in js
    assert "\n" not in js


def test_score_rows_reads_the_raters_vote_and_flags_new_votes():
    key = pd.DataFrame({"packet_id": [1, 2, 3], "label_id": ["10", "11", "12"], "cell": ["contested", "decoy", "missed"],
                        "verdict": ["incorrect", "correct", "correct"], "true": ["correct", "correct", "incorrect"],
                        "prior_rater_vote": ["", "", "Agree"]})
    after = pd.DataFrame({"label_id": ["10", "11", "12", "13"],
                          "validations": [json.dumps([vote("x", "Agree"), vote("r", "Disagree")]),
                                          json.dumps([vote("x", "Agree")]),
                                          json.dumps([vote("r", "Agree")]), "[]"]})
    rows = rl.score_rows(key, after, "r")
    assert rows.verdict_rater.tolist() == ["incorrect", "", "correct"]
    assert rows.new_vote.tolist() == [True, False, False]
    md = rl.score_markdown(rows)
    assert "| contested | 1 | 1 | 0 | 1/1 = 100.0%" in md and "| decoy | 1 | 0 |" in md
