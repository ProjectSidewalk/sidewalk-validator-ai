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


# --- --pool all: the full-pool ranked Disagree list (issue #80) -------------------------------------------------------

def _rank_pool():
    pool = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3", "c:4", "c:5", "c:6"],
                         "verdict": ["incorrect"] * 5 + ["correct"],
                         "p_incorrect": [0.999, 0.7, 0.7, 0.95, 1.0, 0.1],
                         "tier": ["core", "band", "band", "core", "core", "band"]})
    pool["p_correct"] = 1 - pool.p_incorrect
    pool.loc[pool.label_uid == "c:5", "p_correct"] = 1e-9  # saturated p_incorrect: the log-odds still orders it
    return pool


def test_rank_disagrees_band_first_then_log_odds_then_uid():
    r = rl.rank_disagrees(_rank_pool())
    assert r.label_uid.tolist() == ["c:2", "c:3", "c:5", "c:1", "c:4"]  # band (tied score: by uid), then core
    assert r.rank_position.tolist() == [1, 2, 3, 4, 5]
    assert rl.rank_disagrees(_rank_pool(), "score").label_uid.tolist() == ["c:5", "c:1", "c:4", "c:2", "c:3"]
    with pytest.raises(ValueError, match="rank must be"):
        rl.rank_disagrees(_rank_pool(), "random")


def test_pick_all_caps_samples_decoys_and_interleaves():
    n = 400
    pool = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(n)], "verdict": ["incorrect"] * 100 + ["correct"] * 300,
                         "p_incorrect": [0.99 - i * 1e-4 for i in range(100)] + [0.1] * 300, "tier": "core",
                         "true": ""})
    pool["p_correct"] = 1 - pool.p_incorrect
    p = rl.pick_all(pool, "s", cap=40, n_decoys=40)
    assert p.cell.value_counts().to_dict() == {"ranked": 40, "decoy": 40}
    assert set(p[p.cell == "ranked"].rank_position) == set(range(1, 41))  # the top 40, not a sample
    assert set(p[p.cell == "ranked"].label_uid) == {f"c:{i}" for i in range(40)}
    pos = p[p.cell == "decoy"].packet_id
    assert pos.min() <= 5 and pos.max() >= 75 and 30 <= pos.median() <= 50  # not front-loaded
    with pytest.raises(ValueError, match="exceeds the 500-label Gallery list"):
        rl.pick_all(pool, "s", cap=460, n_decoys=50)


def _all_ai_run(tmp_path, all_ai=True):
    """A select --all-ai run of 14 AI labels in x-yy: 0-5 Disagree verdicts (0-2 band), 6-13 Agree; 3, 4 and 10 are
    unvoted; the rater voted on 1 and 7."""
    run = tmp_path / "run"
    (run / "rawlabels").mkdir(parents=True)
    n = 14
    uids = [f"x-yy:{i}" for i in range(n)]
    p_inc = [0.9, 0.95, 0.8, 0.99, 0.6, 0.97] + [0.05] * 8
    unvoted = {3, 4, 10}
    pd.DataFrame({"label_uid": uids, "city": "x-yy", "label_id": [str(i) for i in range(n)], "pano_id": "p",
                  "author": "ai", "pano_source": "gsv", "time_created": "",
                  "tier": ["band"] * 3 + ["core"] * 11, "human_agree": [0 if i in unvoted else 1 for i in range(n)],
                  "human_disagree": 0, "human_unsure": 0, "n_votes": [0 if i in unvoted else 1 for i in range(n)],
                  "label_class": ["" if i in unvoted else "correct" for i in range(n)], "expert_agree": 0,
                  "expert_disagree": 0, "expert_class": "", "round3_split": "", "clean": 1, "severity": "", "tags": "",
                  "pano_x": 1, "pano_y": 1, "pano_width": 2, "pano_height": 1, "camera_pitch": "", "camera_roll": ""}
                 ).to_csv(run / "labels.csv", index=False)
    pd.DataFrame({"label_uid": uids, "status": "ok", "depth_m": 1, "crop_half_px": 1, "tilt_applied": 0,
                  "pose_unhandled": 0}).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame({"label_uid": uids, "p_incorrect": p_inc, "p_correct": [1 - p for p in p_inc]}
                 ).to_csv(run / "predictions.csv", index=False)
    rater = "rater-0000-uuid"
    val = ["[]" if i in unvoted else json.dumps([vote(rater if i in (1, 7) else "c", "Agree")]) for i in range(n)]
    pd.DataFrame({"label_id": [str(i) for i in range(n)], "validations": val}).to_csv(run / "rawlabels" / "x-yy.csv",
                                                                                        index=False)
    json.dump({"cities": [{"city_id": "x-yy", "url": "https://x/"}]}, open(run / "rawlabels" / "cities.json", "w"))
    json.dump({"all_ai": all_ai}, open(run / "selection.json", "w"))
    return run, rater, val


def test_build_pool_all_ranks_disagrees_excludes_rater_votes_and_keeps_the_key_private(tmp_path, monkeypatch):
    run, rater, _ = _all_ai_run(tmp_path)
    monkeypatch.setenv("RATER", rater)
    monkeypatch.setenv("EXPERTS", "")
    out = tmp_path / "ranked"
    rl.main(["build", "--run", str(run), "--city", "x-yy", "--out", str(out), "--pool", "all", "--cap", "3",
             "--n-decoys", "2"])
    key = pd.read_csv(out / "private-key-do-not-share" / "key.csv", keep_default_na=False)
    assert "x-yy:1" not in set(key.label_uid) and "x-yy:7" not in set(key.label_uid)
    ranked = key[key.cell == "ranked"].sort_values("rank_position")
    assert ranked.label_uid.tolist() == ["x-yy:0", "x-yy:2", "x-yy:3"]  # band 0, 2 (1 is the rater's), then core 0.99
    assert {"rank_position", "rank_score", "n_votes", "prior_voters", "prior_status", "tier"} <= set(key.columns)
    assert ranked.set_index("label_uid").loc["x-yy:3", "prior_status"] == "unvoted"
    assert (key.cell == "decoy").sum() == 2
    man = json.load(open(out / "manifest.json"))
    assert man["pool"] == "all" and man["cap"] == 3 and man["rank"] == "band_first" and man["n_ranked"] == 3
    assert man["n_unvoted_ranked"] == 1 and man["excluded_rater_voted"] == 2 and man["n_disagree_in_pool"] == 5
    assert sorted(p.name for p in out.iterdir()) == ["gallery-mask.css", "gallery_url.txt", "index.html",
                                                    "manifest.json", "private-key-do-not-share"]
    public = json.dumps(man) + open(out / "index.html").read() + open(out / "gallery_url.txt").read()
    assert rater not in public


def test_build_pool_all_refuses_a_run_not_selected_with_all_ai(tmp_path, monkeypatch):
    run, rater, _ = _all_ai_run(tmp_path, all_ai=False)
    monkeypatch.setenv("RATER", rater)
    with pytest.raises(SystemExit, match="select --all-ai"):
        rl.main(["build", "--run", str(run), "--city", "x-yy", "--out", str(tmp_path / "o"), "--pool", "all"])
    with pytest.raises(SystemExit, match="exceeds the 500-label"):
        rl.main(["build", "--run", str(run), "--city", "x-yy", "--out", str(tmp_path / "o"), "--pool", "all",
                 "--cap", "480", "--n-decoys", "50"])
    assert not (tmp_path / "o").exists()


def test_score_pool_all_reports_the_hit_rate(tmp_path, monkeypatch):
    run, rater, val = _all_ai_run(tmp_path)
    monkeypatch.setenv("RATER", rater)
    monkeypatch.setenv("EXPERTS", "")
    out = tmp_path / "ranked"
    rl.main(["build", "--run", str(run), "--city", "x-yy", "--out", str(out), "--pool", "all", "--cap", "4",
             "--n-decoys", "2"])
    key = pd.read_csv(out / "private-key-do-not-share" / "key.csv", dtype={"label_id": str})
    votes = {lid: ("Disagree" if cell == "ranked" and i % 2 == 0 else "Agree")
             for i, (lid, cell) in enumerate(zip(key.label_id, key.cell))}
    after = []
    for i in range(14):
        v = json.loads(val[i])
        if str(i) in votes:
            v.append(vote(rater, votes[str(i)]))
        after.append(json.dumps(v))
    (out / "rawlabels_after").mkdir()
    pd.DataFrame({"label_id": [str(i) for i in range(14)], "validations": after}).to_csv(
        out / "rawlabels_after" / "x-yy.csv", index=False)
    rl.main(["score", "--run", str(run), "--out", str(out)])
    md = (out / "results.md").read_text()
    n_hits = sum(1 for lid, c in zip(key.label_id, key.cell) if c == "ranked" and votes[lid] == "Disagree")
    assert f"| ranked, all | 4 | 4 | 0 | {n_hits}/4" in md
    assert "| ranked, tier band |" in md and "| ranked, unvoted |" in md and "| ranked, rank 1-50 | 4 |" in md
    assert "| decoys (validator Agree) | 2 | 2 | 0 | 0/2" in md and "38 % [29, 49]" in md
    assert json.load(open(out / "manifest.json"))["scored"]["readback"]["gained_rater_vote"] == 6


def test_hit_rate_markdown_excludes_unsure_from_the_denominator():
    rows = pd.DataFrame({"cell": ["ranked"] * 3 + ["decoy"], "tier": ["band", "band", "core", "core"],
                         "prior_status": ["unvoted", "other-voted", "unvoted", "unvoted"],
                         "rank_position": [1, 2, 51, None],
                         "verdict_rater": ["incorrect", "cant_tell", "correct", "correct"]})
    md = rl.hit_rate_markdown(rows)
    assert "| ranked, all | 3 | 3 | 1 | 1/2 = 50.0%" in md
    assert "| ranked, rank 1-50 | 2 |" in md and "| ranked, rank 51-100 | 1 |" in md
    assert "| ranked, tier band | 2 | 2 | 1 | 1/1" in md


def test_pick_all_decoys_are_agree_verdicts_disjoint_from_the_ranked_rows():
    n = 400
    pool = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(n)], "verdict": ["incorrect"] * 100 + ["correct"] * 300,
                         "p_incorrect": [0.99 - i * 1e-4 for i in range(100)] + [0.1] * 300, "tier": "core",
                         "true": ""})
    pool["p_correct"] = 1 - pool.p_incorrect
    p = rl.pick_all(pool, "s", cap=40, n_decoys=60)
    decoys, ranked = p[p.cell == "decoy"], p[p.cell == "ranked"]
    assert (decoys.verdict == "correct").all() and len(decoys) == 60
    assert not set(decoys.label_uid) & set(ranked.label_uid) and p.label_uid.is_unique
