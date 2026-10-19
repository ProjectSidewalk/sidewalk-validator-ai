"""validator_reliability.py (issue #12): leave-one-out reliability, the vote filters, and vote_weights.csv."""
import doctest
import json

import pandas as pd
import pytest

import validator_reliability as vr


def _cell(*votes, ai=()):
    """A rawLabels ``validations`` cell: ``votes`` are ``(user_id, "Agree"|"Disagree"|"Unsure")`` human votes."""
    out = [{"user_id": u, "validation": v, "validator_type": "Human"} for u, v in votes]
    out += [{"user_id": u, "validation": v, "validator_type": "AI"} for u, v in ai]
    return json.dumps(out)


def test_doctests():
    result = doctest.testmod(vr)
    assert result.attempted > 0 and result.failed == 0


def test_unsure_and_ai_votes_are_ignored():
    votes, _ = vr.human_votes(_cell(("a", "Agree"), ("b", "Unsure"), ai=[("bot", "Disagree")]))
    assert votes == {"a": True}
    # A missing validator_type fails closed, like data_generation.human_vote_counts.
    votes, _ = vr.human_votes('[{"user_id": "x", "validation": "Agree"}]')
    assert votes == {}


def test_duplicate_voter_keeps_last_vote():
    votes, dup = vr.human_votes(_cell(("a", "Agree"), ("b", "Agree"), ("a", "Disagree")))
    assert votes == {"b": True, "a": False} and dup == 1


def test_leave_one_out_excludes_own_vote():
    # 3 Agree + 1 Disagree. For d, the others are 3-0 Agree: a miss. If d's own vote counted, it would not matter
    # here, but on 2-2 + d the own vote would break the tie; leave-one-out keeps it a tie (not scored).
    labels = [vr.human_votes(_cell(("a", "Agree"), ("b", "Agree"), ("c", "Agree"), ("d", "Disagree")))[0]]
    c = vr.reliability_counts(labels)
    assert c == {"a": [1, 1], "b": [1, 1], "c": [1, 1], "d": [0, 1]}
    tie = [{"a": True, "b": True, "c": False, "d": False, "e": True}]
    assert vr.reliability_counts(tie)["e"] == [0, 0]   # others 2-2
    assert vr.reliability_counts(tie)["c"] == [0, 1]   # others 3-1 Agree


def test_needs_three_other_votes_and_unsure_does_not_count_toward_them():
    # Two Agree/Disagree others plus an Unsure: only 2 counted others, so nothing is scored.
    votes, _ = vr.human_votes(_cell(("a", "Agree"), ("b", "Agree"), ("c", "Agree"), ("u", "Unsure"),
                                    ai=[("bot", "Agree")]))
    assert vr.reliability_counts([votes]) == {"a": [0, 0], "b": [0, 0], "c": [0, 0]}
    assert vr.reliability_counts([votes], min_other_votes=2)["a"] == [1, 1]


def test_smoothing_prior_and_unscored_validators():
    table, m = vr.smooth({"a": [10, 10], "b": [0, 0]}, prior_strength=10)
    assert m == 1.0 and table["b"]["reliability"] == 1.0 and table["b"]["n"] == 0
    table, m = vr.smooth({"a": [1, 1], "b": [0, 1]}, prior_strength=2)
    assert m == 0.5 and table["a"]["reliability"] == pytest.approx(2 / 3) and table["a"]["raw_rate"] == 1.0
    with pytest.raises(ValueError, match="no scored votes"):
        vr.smooth({"a": [0, 0]})


def test_main_writes_outputs_keyed_on_city_and_label_id(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    five = _cell(("a", "Agree"), ("b", "Agree"), ("c", "Agree"), ("d", "Agree"), ("e", "Disagree"))
    # The same label_id in two cities must stay two labels (label_id is a per-city serial).
    pd.DataFrame({"label_id": [7, 8], "validations": [five, None]}).to_csv(raw / "city-a.csv", index=False)
    pd.DataFrame({"label_id": [7], "validations": [_cell(("a", "Disagree"))]}).to_csv(raw / "city-b.csv",
                                                                                         index=False)
    (raw / "cities.json").write_text("[]")   # non-CSV files in the cache are ignored
    out = tmp_path / "out"
    vr.main(["--rawlabels", str(raw), "--out", str(out), "--prior-strength", "0", "--gold-min-reliability", "1.0"])

    rel = json.loads((out / "validator_reliability.json").read_text())
    assert rel["a"] == {"reliability": 1.0, "n": 1, "hits": 1, "raw_rate": 1.0}
    assert rel["e"]["reliability"] == 0.0
    w = pd.read_csv(out / "vote_weights.csv", dtype={"label_uid": str})
    assert list(w.columns) == vr.VOTE_WEIGHT_COLUMNS
    assert w.label_uid.tolist() == ["city-a:7", "city-b:7"]   # city-a:8 has no votes
    a7 = w.iloc[0]
    assert (a7.w_agree, a7.w_disagree, a7.n_votes, a7.n_gold, a7.min_reliability) == (4.0, 0.0, 5, 4, 0.0)
    b7 = w.iloc[1]
    assert (b7.w_agree, b7.w_disagree, b7.n_votes, b7.n_gold) == (0.0, 1.0, 1, 1)
    run = json.loads((out / "reliability_run.json").read_text())
    assert run["files"] == ["city-a.csv", "city-b.csv"] and run["prior_mean"] == pytest.approx(0.8)
