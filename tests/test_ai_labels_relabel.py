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
                        "p_incorrect": [0.999, 0.01, 0.02], "prior_rater_vote": ["", "", "Agree"]})
    after = pd.DataFrame({"label_id": ["10", "11", "12", "13"],
                          "validations": [json.dumps([vote("x", "Agree"), vote("r", "Disagree")]),
                                          json.dumps([vote("x", "Agree")]),
                                          json.dumps([vote("r", "Agree")]), "[]"]})
    rows = rl.score_rows(key, after, "r")
    assert rows.verdict_rater.tolist() == ["incorrect", "", "correct"]
    assert rows.new_vote.tolist() == [True, False, False]
    md = rl.score_markdown(rows)
    assert "| contested | 1 | 1 | 0 | 1/1 = 100.0%" in md and "| decoy | 1 | 0 |" in md


def test_pick_interleaves_decoys_instead_of_front_loading_them():
    pool = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(400)], "verdict": ["incorrect"] * 40 + ["correct"] * 360,
                         "true": "correct"})
    p = rl.pick(pool, "s", n_decoys=40)
    pos = p[p.cell == "decoy"].packet_id
    # 40 decoys among 80 rows: a front-loaded list would put them all in 1..40 (median 20.5)
    assert pos.min() <= 5 and pos.max() >= 75 and 30 <= pos.median() <= 50
    # and a different seed gives a different order of the same labels
    q = rl.pick(pool, "t", n_decoys=40)
    assert set(q.label_uid) != set(p.label_uid) or q.label_uid.tolist() != p.label_uid.tolist()


def test_score_markdown_excludes_unsure_and_folds_in_prior_votes():
    rows = pd.DataFrame({"packet_id": [1, 2, 3, 4], "cell": ["contested", "contested", "confirmed", "decoy"],
                         "verdict": ["incorrect", "incorrect", "incorrect", "correct"],
                         "true": ["correct", "correct", "incorrect", "correct"],
                         "p_incorrect": [0.9999, 0.995, 0.9999, 0.01],
                         "verdict_rater": ["cant_tell", "correct", "incorrect", "correct"],
                         "rater_vote": ["Unsure", "Agree", "Disagree", "Agree"],
                         "prior_rater_vote": ["", "", "", ""], "new_vote": [True] * 4})
    md = rl.score_markdown(rows)
    assert "| contested | 2 | 2 | 1 | 0/1 = 0.0%" in md          # the Unsure is not a decided vote
    assert "packet only (contested + confirmed): 1/2 = 50.0%" in md
    prior = pd.DataFrame({"label_uid": ["c:9", "c:8"], "p_incorrect": [0.9999, 0.5], "rater_vote": ["Agree", "Unsure"]})
    md2 = rl.score_markdown(rows, prior)
    assert "over all 5 Disagrees (3 decided) 1/3 = 33.3%" in md2   # 3 packet + 2 prior; Unsure twice excluded


def test_readback_check_rejects_foreign_or_vanished_votes():
    key = pd.DataFrame({"label_id": ["1"]})
    b = pd.DataFrame({"label_id": ["1"], "validations": [json.dumps([vote("x", "Agree")])]})
    other = pd.DataFrame({"label_id": ["1"], "validations": [json.dumps([vote("x", "Agree"), vote("y", "Agree")])]})
    with pytest.raises(ValueError):
        rl.readback_check(key, b, other, "r")
    two = pd.DataFrame({"label_id": ["1"], "validations": [json.dumps([vote("x", "Agree"), vote("r", "Agree"), vote("r", "Agree")])]})
    with pytest.raises(ValueError):
        rl.readback_check(key, b, two, "r")
    dup = pd.DataFrame({"label_id": ["1", "1"], "validations": ["[]", "[]"]})
    with pytest.raises(ValueError):
        rl.readback_check(key, dup, dup, "r")


def test_build_excludes_rater_voted_labels_and_keeps_the_key_private(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "rawlabels").mkdir(parents=True)
    n = 12
    uids = [f"x-yy:{i}" for i in range(n)]
    verdict_p = [0.99] * 4 + [0.01] * 8            # 4 validator Disagrees, 8 Agrees
    true = ["correct", "incorrect", "correct", "correct", "incorrect", "correct", "correct", "correct", "correct",
            "correct", "correct", "correct"]
    pd.DataFrame({"label_uid": uids, "city": "x-yy", "label_id": [str(i) for i in range(n)], "pano_id": "p",
                  "author": "ai", "pano_source": "gsv", "time_created": "", "tier": "core", "human_agree": 1,
                  "human_disagree": 0, "human_unsure": 0, "n_votes": 1, "label_class": true, "expert_agree": 0,
                  "expert_disagree": 0, "expert_class": "", "round3_split": "", "clean": 1, "severity": "", "tags": "",
                  "pano_x": 1, "pano_y": 1, "pano_width": 2, "pano_height": 1, "camera_pitch": "", "camera_roll": ""}
                 ).to_csv(run / "labels.csv", index=False)
    pd.DataFrame({"label_uid": uids, "status": "ok", "depth_m": 1, "crop_half_px": 1, "tilt_applied": 0,
                  "pose_unhandled": 0}).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame({"label_uid": uids, "p_incorrect": verdict_p, "p_correct": [1 - p for p in verdict_p]}
                 ).to_csv(run / "predictions.csv", index=False)
    # the rater voted on label 0 (a contested one) and label 5 (a decoy candidate); label 4 is a missed one
    rater = "rater-0000-uuid"
    val = [json.dumps([vote(rater if i in (0, 5) else "c", "Agree")]) for i in range(n)]
    pd.DataFrame({"label_id": [str(i) for i in range(n)], "validations": val}).to_csv(run / "rawlabels" / "x-yy.csv", index=False)
    json.dump({"cities": [{"city_id": "x-yy", "url": "https://x/"}]}, open(run / "rawlabels" / "cities.json", "w"))
    monkeypatch.setenv("RATER", rater)
    monkeypatch.setenv("EXPERTS", "")
    out = tmp_path / "relabel"
    rl.main(["build", "--run", str(run), "--city", "x-yy", "--out", str(out), "--n-decoys", "3"])
    key = pd.read_csv(out / "private-key-do-not-share" / "key.csv")
    assert "x-yy:0" not in set(key.label_uid) and "x-yy:5" not in set(key.label_uid)
    assert key.cell.value_counts().to_dict() == {"decoy": 3, "contested": 2, "confirmed": 1, "missed": 1}
    man = json.load(open(out / "manifest.json"))
    assert man["excluded_rater_voted"] == 2 and not man["rawlabels_before"].startswith("/")
    public = man | {"index": open(out / "index.html").read(), "url": open(out / "gallery_url.txt").read()}
    assert "rater_id" not in man and rater not in json.dumps(public)
    assert rater in open(out / "private-key-do-not-share" / "key.csv").read()
    assert open(out / "gallery_url.txt").read().startswith("https://x/gallery?labelIds=")
