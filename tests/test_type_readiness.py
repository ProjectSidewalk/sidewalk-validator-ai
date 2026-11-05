"""analysis/type_readiness.py and calibration/nosidewalk_prior.py on synthetic rawLabels (issue #42)."""
import doctest
import json

import pandas as pd
import pytest

import data_generation as dg
from analysis import type_readiness as tr
from calibration import nosidewalk_prior as npr
from test_nosidewalk_faces import EXPERT, ns_rows, votes

LISTING = {"cities": [{"city_id": "pub-a", "visibility": "public", "url": "https://sidewalk-a.cs.washington.edu"},
                      {"city_id": "pub-b", "visibility": "public", "url": "https://sidewalk-b.cs.washington.edu"},
                      {"city_id": "secret", "visibility": "private", "url": None}]}


def city_frame(city, base_edge):
    """One city's NoSidewalk rows: a face decided by two validators (3 labels), a face where one validator voted
    three times (the census counts it, training does not), an incorrect face, an expert face, a margin label, and
    a label without a face."""
    spec = [(base_edge, "left", votes(("a", "Agree")), "l1"), (base_edge, "left", votes(("b", "Agree")), "l1"),
            (base_edge, "left", "[]", "l1"),
            (base_edge + 1, "left", votes(("a", "Agree")), "l2"), (base_edge + 1, "left", votes(("a", "Agree")), "l2"),
            (base_edge + 2, "right", votes(("a", "Disagree")), "l3"),
            (base_edge + 2, "right", votes(("b", "Disagree")), "l3"),
            (base_edge + 3, "right", votes((EXPERT, "Disagree")), "l4"), (base_edge + 3, "right", "[]", "l4"),
            (base_edge + 4, "left", votes(("a", "Agree"), ("b", "Agree"), ("c", "Agree")), "l5"),
            (None, None, votes(("a", "Agree"), ("b", "Agree")), "l6")]
    df = ns_rows(spec, city=city)
    df.loc[9, "tags"] = '["street has no sidewalks"]'  # the margin label
    return df


def write_cache(tmp_path, pattern="{city}.csv"):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i, city in enumerate(("pub-a", "pub-b", "secret")):
        df = city_frame(city, 10 * i + 1)
        df.drop(columns="city").to_csv(cache / pattern.format(city=city, label_type="NoSidewalk"), index=False)
    (cache / "cities.json").write_text(json.dumps({**LISTING, "fetched_on": "2026-11-05"}))
    listing = tmp_path / "listing.json"
    listing.write_text(json.dumps(LISTING))
    return cache, listing


def test_load_cache_reads_listed_cities_and_both_layouts(tmp_path):
    cache, _ = write_cache(tmp_path, "{city}__{label_type}.csv")
    rows, cities = tr.load_cache(str(cache), "NoSidewalk", "{city}__{label_type}.csv")
    assert cities == ["pub-a", "pub-b", "secret"] and len(rows) == 33
    (cache / "cities.json").unlink()  # file names alone
    assert tr.load_cache(str(cache), "NoSidewalk", "{city}__{label_type}.csv")[1] == ["pub-a", "pub-b", "secret"]
    with pytest.raises(SystemExit):
        tr.load_cache(str(cache), "Signal", "{city}__{label_type}.csv")


def test_readiness_counts(tmp_path):
    cache, _ = write_cache(tmp_path)
    rows, _ = tr.load_cache(str(cache), "NoSidewalk")
    public = tr.public_cities(LISTING)
    r = tr.readiness(rows, "NoSidewalk", frozenset({EXPERT}), public, cap=2)
    assert r["labels"] == 33 and r["cities"] == 3 and r["n_experts"] == 1
    # margin: per city 1 correct (3 agree) and 0 incorrect (each Disagree label has one vote)
    assert r["label_rule_margin"]["total"] == {"correct": 3}
    # experts: + the expert's own label per city
    assert r["label_rule_experts"]["class_by_source"] == {"correct": {"margin": 3}, "incorrect": {"expert": 3}}
    cen = r["face_rule_census"]
    # census (vote sums, every label on the face): faces 1, 2 (one person twice) and 5 correct; 3 and 4 incorrect
    assert cen["correct"] == {"faces": 9, "labels": 18, "single_labeler_share": 1.0}
    assert cen["incorrect"]["faces"] == 6 and cen["incorrect"]["labels"] == 12
    assert {d["city"] for d in cen["incorrect_labels_by_city"]} == {"pub-a", "pub-b", tr.PRIVATE}
    pipe = r["face_rule_pipeline"]["before_cap"]
    # training: face 2 is not decided (one validator); face 1's three labels by face_margin
    assert pipe["class_by_source"]["correct"] == {"face_margin": 9, "margin": 3}
    assert pipe["class_by_source"]["incorrect"] == {"expert": 3, "face_expert": 3, "face_margin": 6}
    after = r["face_rule_pipeline"]["after_cap"]
    assert after["total"] == {"correct": 9, "incorrect": 12}  # face 1 capped from 3 to 2 per city
    assert r["vote_pairs"][0] == {"agree": 1, "disagree": 0, "n": 12}
    assert r["faces"]["labels_with_face"] == 30 and r["faces"]["faces"] == 15
    assert r["faces"]["faces_with_both_classes"] == 0
    assert r["tags_by_class"] == {"<none>": {"incorrect": 3}, "street has no sidewalks": {"correct": 3}}
    assert r["face_size_prior"]["face_size_shares"] == {"1": 0.1, "2": 0.6, "3": 0.3}


def test_readiness_without_experts_has_no_expert_rows(tmp_path):
    cache, _ = write_cache(tmp_path)
    rows, _ = tr.load_cache(str(cache), "NoSidewalk")
    r = tr.readiness(rows, "NoSidewalk", frozenset(), tr.public_cities(LISTING))
    srcs = {s for c in r["face_rule_pipeline"]["before_cap"]["class_by_source"].values() for s in c}
    assert srcs == {"margin", "face_margin"}


def test_main_writes_aggregates_only_and_pools_private_cities(tmp_path, monkeypatch):
    cache, listing = write_cache(tmp_path)
    monkeypatch.setenv("EXPERTS", f"e={EXPERT}")
    out_json, out_md = tmp_path / "out" / "r.json", tmp_path / "out" / "r.md"
    tr.main(["--cache", str(cache), "--label-type", "NoSidewalk", "--listing", str(listing),
             "--out-json", str(out_json), "--out-md", str(out_md)])
    text = out_json.read_text() + out_md.read_text()
    assert EXPERT not in text and "secret" not in text and "l1" not in text
    assert tr.PRIVATE in text
    report = json.loads(out_json.read_text())
    assert report["cache_date"] == "2026-11-05" and report["listing"] == "saved" and report["n_experts"] == 1
    md = out_md.read_text()
    assert "| label margin + expert (1 experts) | 3 | 3 | 0 |" in md
    assert "face rule, census count" in md and "Face-size deployment prior" in md


def test_main_fetches_the_live_listing(tmp_path):
    cache, _ = write_cache(tmp_path)
    seen = []
    tr.main(["--cache", str(cache), "--label-type", "NoSidewalk", "--out-json", str(tmp_path / "r.json")],
            fetch=lambda url: seen.append(url) or LISTING)
    assert seen == [dg.CITIES_API]


def test_face_size_prior_entry_and_cli(tmp_path):
    cache, _ = write_cache(tmp_path)
    priors = tmp_path / "deployment_priors.json"
    priors.write_text(json.dumps({"CurbRamp": {"correct": 1, "incorrect": 1}, "NoSidewalk": {"correct": 0,
                                                                                            "incorrect": 1}}))
    entry = npr.main(["--cache", str(cache), "--priors", str(priors)])
    # 30 labels with a face: 3 alone on a face, 18 on 2-label faces, 9 on 3-label faces
    assert entry["face_size_shares"] == {"1": 0.1, "2": 0.6, "3": 0.3}
    assert entry["p_correct"] == pytest.approx(0.1 * 0.69 + 0.6 * 0.80 + 0.3 * 0.84)
    assert (entry["correct"], entry["incorrect"]) == (24, 6)
    assert entry["labels_without_face"] == 3 and entry["measured_on"] == "2026-11-05" and entry["method"] == "face_size"
    saved = json.loads(priors.read_text())
    assert saved["CurbRamp"] == {"correct": 1, "incorrect": 1} and saved["NoSidewalk"]["correct"] == 24
    assert "5222" in saved["NoSidewalk"]["source"]


def test_face_size_prior_needs_faces():
    with pytest.raises(ValueError):
        npr.prior_entry(pd.DataFrame({"face_id": ["", ""]}), ["x"], "2026-11-05")


def test_ai_labels_do_not_enter_the_prior():
    e = npr.prior_entry(pd.DataFrame({"face_id": ["a", "b", "b"], "user_id": ["u", dg.AI_USER_ID, "u"]}), ["x"], "d")
    assert e["face_size_shares"] == {"1": 1.0, "2": 0.0, "3": 0.0}


@pytest.mark.parametrize("module", [tr, npr])
def test_doctests(module):
    result = doctest.testmod(module)
    assert result.failed == 0
