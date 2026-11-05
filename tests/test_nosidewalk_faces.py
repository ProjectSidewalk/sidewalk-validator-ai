"""Block-face verdicts for NoSidewalk (issue #42): nosidewalk_faces.py and its hooks in data_generation.py."""
import doctest
import json

import pandas as pd
import pytest

import data_generation as dg
import nosidewalk_faces as nf

AI = dg.AI_USER_ID
EXPERT = "expert-1"


def votes(*vs, ai=()):
    """``votes(("a", "Agree"), ("b", "Disagree"))``: human votes, plus AI votes by direction."""
    v = [{"user_id": u, "validation": x, "validator_type": "Human"} for u, x in vs]
    v += [{"user_id": AI, "validation": x, "validator_type": "AI"} for x in ai]
    return json.dumps(v)


def ns_rows(spec, city="toy-city"):
    """NoSidewalk rows from ``[(edge, side, validations, labeler), ...]``; label_id = position + 1."""
    rows = []
    for i, (edge, side, vals, labeler) in enumerate(spec, start=1):
        rows.append({"label_id": i, "user_id": labeler, "pano_id": f"p{i}", "pano_source": "gsv",
                     "label_type": "NoSidewalk", "validations": vals, "pano_x": 100, "pano_y": 1000,
                     "pano_width": 8192, "pano_height": 4096, "street_edge_id": edge, "street_side": side,
                     "camera_pitch": 0.0, "camera_roll": None, "severity": None, "tags": "[]", "city": city,
                     "latitude": 47.0 + i * 1e-3, "longitude": -122.0})
    return pd.DataFrame(rows)


def decide(rows, experts=frozenset(), **kw):
    eligible, _ = dg.eligible_labels(rows, "NoSidewalk", experts)
    out = nf.add_face_verdicts(rows, eligible, "NoSidewalk", experts, **kw)
    return out.set_index("label_id")[["label_class", "class_source", "face_id"]]


def test_face_id_requires_edge_and_side_and_keeps_the_city():
    d = pd.DataFrame({"city": ["a", "b", "a", "a", "a"], "street_edge_id": [3, 3, "x", 4.0, None],
                      "street_side": ["left", "left", "left", "middle", "right"]})
    assert nf.face_id(d).tolist() == ["a:3:left", "b:3:left", "", "", ""]
    assert nf.face_id(pd.DataFrame({"city": ["a"]})).tolist() == [""]  # no face columns at all


def test_two_distinct_validators_on_different_labels_decide_the_whole_face():
    rows = ns_rows([(1, "left", votes(("a", "Agree")), "l1"), (1, "left", votes(("b", "Agree")), "l1"),
                    (1, "left", "[]", "l1"), (2, "left", votes(("a", "Agree")), "l2")])
    out = decide(rows)
    assert out.loc[[1, 2, 3], "label_class"].tolist() == ["correct"] * 3
    assert set(out.loc[[1, 2, 3], "class_source"]) == {"face_margin"}
    assert 4 not in out.index  # one validator on face 2: nothing


def test_one_validator_on_many_labels_is_one_piece_of_evidence():
    rows = ns_rows([(1, "left", votes(("a", "Agree")), "l1"), (1, "left", votes(("a", "Agree")), "l1"),
                    (1, "left", votes(("a", "Agree")), "l1")])
    assert decide(rows).empty
    v = nf.face_votes(rows.assign(face_id=nf.face_id(rows)))
    assert v.loc["toy-city:1:left", ["support", "agree_votes"]].tolist() == [1, 3]


def test_incorrect_face_and_contradicted_face():
    rows = ns_rows([(1, "right", votes(("a", "Disagree")), "l1"), (1, "right", votes(("b", "Disagree")), "l1"),
                    (2, "right", votes(("a", "Agree"), ("b", "Agree")), "l1"),
                    (2, "right", votes(("c", "Disagree")), "l1"), (2, "right", "[]", "l1")])
    out = decide(rows)
    assert out.loc[[1, 2], "label_class"].tolist() == ["incorrect", "incorrect"]
    assert 3 not in out.index and 5 not in out.index  # contradicted face 2 decides nothing


def test_label_rule_wins_and_is_never_overridden():
    # label 1 is margin-correct (3 agree); the face also has a lone Disagree on label 2: contradicted face,
    # so label 1 keeps its margin verdict and label 2 stays out.
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree"), ("c", "Agree")), "l1"),
                    (1, "left", votes(("d", "Disagree")), "l1"), (1, "left", "[]", "l1")])
    out = decide(rows)
    assert out.loc[1, "label_class"] == "correct" and out.loc[1, "class_source"] == "margin"
    assert list(out.index) == [1]


def test_label_rule_rows_keep_their_source_on_a_decided_face():
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree"), ("c", "Agree")), "l1"),
                    (1, "left", "[]", "l1")])
    out = decide(rows)
    assert out.loc[1, "class_source"] == "margin" and out.loc[2, "class_source"] == "face_margin"


def test_expert_face_needs_no_contradicting_human():
    rows = ns_rows([(1, "left", votes((EXPERT, "Agree")), "l1"), (1, "left", "[]", "l1"),
                    (2, "left", votes((EXPERT, "Disagree")), "l1"), (2, "left", votes(("c", "Agree")), "l1"),
                    (3, "left", votes((EXPERT, "Disagree")), "l1"), (3, "left", "[]", "l1")])
    out = decide(rows, frozenset({EXPERT}))
    # face 1: the expert's own label is decided by the label rule (expert), the other by the face (face_expert)
    assert out.loc[1, "class_source"] == "expert" and out.loc[2, "class_source"] == "face_expert"
    assert out.loc[2, "label_class"] == "correct"
    # face 2: the expert's lone Disagree decides label 3 by the label rule; c's Agree contradicts the face, so
    # label 4 stays out
    assert out.loc[3, "class_source"] == "expert" and 4 not in out.index
    assert out.loc[6, ["label_class", "class_source"]].tolist() == ["incorrect", "face_expert"]
    assert decide(rows).empty  # without EXPERTS the expert is one validator: nothing


def test_labels_without_a_face_keep_the_label_rule_only():
    rows = ns_rows([(None, "left", votes(("a", "Agree"), ("b", "Agree")), "l1"),
                    (1, None, votes(("a", "Agree"), ("b", "Agree")), "l1"),
                    (2, "left", votes(("a", "Disagree"), ("b", "Disagree")), "l1")])
    out = decide(rows)
    assert list(out.index) == [3] and out.loc[3, "class_source"] == "margin"


def test_ai_votes_never_count():
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ai=("Agree", "Agree")), "l1"), (1, "left", "[]", "l1"),
                    (2, "left", votes(("a", "Agree"), ("b", "Agree"), ai=("Disagree",)), "l1")])
    out = decide(rows)
    assert 1 not in out.index and 2 not in out.index
    assert out.loc[3, "class_source"] == "face_margin"  # the AI's Disagree does not contradict


def test_bad_geometry_rows_neither_decide_nor_get_decided():
    rows = ns_rows([(1, "left", votes(("a", "Agree")), "l1"), (1, "left", votes(("b", "Agree")), "l1"),
                    (1, "left", "[]", "l1")])
    rows.loc[1, "pano_width"] = 0  # label 2 dropped by valid_geometry: its Agree no longer counts
    assert decide(rows).empty


def test_added_rows_have_the_columns_selection_needs():
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree")), "l1"), (1, "left", "[]", AI)])
    eligible, _ = dg.eligible_labels(rows, "NoSidewalk")
    out = nf.add_face_verdicts(rows, eligible, "NoSidewalk")
    assert out.set_index("label_id").loc[2, "author"] == "ai"
    assert out.set_index("label_id").loc[1, ["human_agree", "human_disagree"]].tolist() == [2, 0]
    sel = dg.select_training_labels(out)  # no incorrect: empty but well-formed
    assert len(sel) == 0


def test_cap_keeps_the_lowest_ranks_per_face_and_class_and_is_stable():
    rows = ns_rows([(1, "left", "[]", "l1")] * 8 + [(2, "left", "[]", "l1")] * 2 + [(None, None, "[]", "l1")] * 3)
    rows["face_id"] = nf.face_id(rows)
    rows["label_class"] = ["correct"] * 6 + ["incorrect"] * 2 + ["correct"] * 5
    capped = nf.cap_labels_per_face(rows, 3, seed=42)
    per = capped.groupby(["face_id", "label_class"]).size().to_dict()
    assert per == {("", "correct"): 3, ("toy-city:1:left", "correct"): 3, ("toy-city:1:left", "incorrect"): 2,
                   ("toy-city:2:left", "correct"): 2}
    ranks = sorted((dg.label_rank(42, f"toy-city:{i}"), i) for i in range(1, 7))
    assert set(capped.loc[capped.label_class.eq("correct") & capped.face_id.eq("toy-city:1:left"), "label_id"]) == \
        {i for _, i in ranks[:3]}
    assert capped.equals(nf.cap_labels_per_face(rows.sample(frac=1, random_state=1), 3, seed=42).sort_index())
    # one label leaving the face changes the kept set by at most one
    dropped = rows[rows.label_id != ranks[0][1]]
    kept_after = set(nf.cap_labels_per_face(dropped, 3, seed=42).label_id)
    assert len(set(capped.label_id) - kept_after) == 1
    assert nf.cap_labels_per_face(rows, 0, seed=42) is rows
    with pytest.raises(ValueError):
        nf.cap_labels_per_face(rows, -1, seed=42)


def test_apply_verdict_unit(capsys):
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree")), "l1")] + [(1, "left", "[]", "l1")] * 7)
    eligible, _ = dg.eligible_labels(rows, "NoSidewalk")
    assert nf.apply_verdict_unit(rows, eligible, "NoSidewalk", frozenset(), "label", 5, 42) is eligible
    out = nf.apply_verdict_unit(rows, eligible, "NoSidewalk", frozenset(), "face", 5, 42)
    assert len(out) == 5 and set(out.class_source) == {"face_margin"}
    assert "cap 5 per face and class dropped 3" in capsys.readouterr().out
    with pytest.raises(ValueError):
        nf.apply_verdict_unit(rows, eligible, "NoSidewalk", frozenset(), "street", 5, 42)


def test_deployment_prior_ignores_face_rows():
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree"), ("c", "Agree")), "l1"),
                    (1, "left", "[]", "l1"), (2, "left", votes(("a", "Disagree"), ("b", "Disagree")), "l1"),
                    (2, "left", "[]", "l1")])
    eligible, _ = dg.eligible_labels(rows, "NoSidewalk")
    out = nf.add_face_verdicts(rows, eligible, "NoSidewalk")
    entry = dg.deployment_prior_entry(out, ["toy-city"], len(rows))
    assert (entry["correct"], entry["incorrect"]) == (1, 1)


def test_resume_settings_are_optional_settings():
    assert nf.resume_settings("label", 5) == {}
    assert {"verdict_unit", "max_labels_per_face"} <= set(dg.OPTIONAL_SETTINGS)


def test_check_resume_catches_a_verdict_unit_or_cap_change(tmp_path):
    run = tmp_path / "generation_run.json"
    base = {"tilt_beta": 0.95}
    run.write_text(json.dumps({"selection_sha256": "h", "settings": base}))
    face = {**base, **nf.resume_settings("face", 5)}
    assert dg.check_resume(str(run), "h", face)[1] == {"verdict_unit": (None, "face"), "max_labels_per_face": (None, 5)}
    run.write_text(json.dumps({"selection_sha256": "h", "settings": face}))
    assert dg.check_resume(str(run), "h", face)[1] == {}
    assert dg.check_resume(str(run), "h", {**base, **nf.resume_settings("face", 3)})[1] == \
        {"max_labels_per_face": (5, 3)}
    assert set(dg.check_resume(str(run), "h", base)[1]) == {"verdict_unit", "max_labels_per_face"}


def test_manifest_row_and_resume_refill_face_id(tmp_path):
    base = tmp_path / "Toy"
    (base / "correct").mkdir(parents=True)
    path = str(base / "manifest.csv")
    old_cols = [c for c in dg.MANIFEST_COLUMNS if c != "face_id"]  # a manifest from before face_id existed
    (base / "correct" / "c_1.webp").write_bytes(b"x")
    pd.DataFrame([{"label_uid": "c:1", "city": "c", "label_id": "1", "pano_id": "p", "label_class": "correct",
                   "class_source": "margin", "crop_path": "correct/c_1.webp", "status": "ok",
                   "pano_path": "/store/c/p/p.jpg"}], columns=old_cols).to_csv(path, index=False)
    sel = {"c:1": "correct"}
    with dg.ManifestWriter(path, str(base), sel, sources={"c:1": "face_margin"}, face_ids={"c:1": "c:9:left"}) as m:
        assert m.is_done("c:1")
    kept = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(kept.columns) == dg.MANIFEST_COLUMNS
    assert kept.loc[0, ["face_id", "class_source"]].tolist() == ["c:9:left", "face_margin"]
    rec = dg.manifest_row({"city": "c", "label_id": 2, "pano_id": "p", "label_class": "correct", "author": "human",
                           "human_agree": 0, "human_disagree": 0, "face_id": "c:9:left"}, "NoSidewalk", "x", 0.95)
    assert rec["face_id"] == "c:9:left"


def test_select_only_with_face_unit(tmp_path, capsys):
    rows = ns_rows([(1, "left", votes(("a", "Agree"), ("b", "Agree")), "l1")] + [(1, "left", "[]", "l1")] * 3
                   + [(2, "right", votes(("a", "Disagree"), ("b", "Disagree")), "l2")] + [(2, "right", "[]", "l2")] * 2)
    rows.drop(columns="city").to_csv(tmp_path / "toy-city.csv", index=False)
    (tmp_path / dg.CITIES_CACHE_FILE).write_text(json.dumps(
        {"cities": [{"city_id": "toy-city", "visibility": "public", "url": "https://sidewalk-toy.cs.washington.edu"}]}))
    common = ["--label_type", "NoSidewalk", "--select-only", "--rawlabels-cache", str(tmp_path)]
    dg.main(common + ["--out", str(tmp_path / "label")])
    assert "--select-only: 1 labels selected" in capsys.readouterr().out  # label 5: two Disagrees, margin rule
    dg.main(common + ["--out", str(tmp_path / "face"), "--verdict-unit", "face", "--max-labels-per-face", "3"])
    out = capsys.readouterr().out
    assert "1 label-rule labels + 6 face-rule labels" in out
    assert "cap 3 per face and class dropped 1" in out
    # 3 incorrect -> int(0.55/0.45 * 3) = 3 correct
    assert "--select-only: 6 labels selected" in out
    prior = json.loads((tmp_path / "face" / "deployment_priors.json").read_text())["NoSidewalk"]
    assert (prior["correct"], prior["incorrect"]) == (0, 1)  # face rows never enter the margin prior


@pytest.mark.parametrize("module", [nf])
def test_doctests(module):
    result = doctest.testmod(module)
    assert result.attempted > 0 and result.failed == 0
