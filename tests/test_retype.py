"""retype.py (issue #13): type edits, the inherited multi-class split, colocation, weights, suggestions, the binary
export to evaluate.py / calibrate.py, the paired comparison with round 3, the report's aggregates and the job files.

Every fixture is synthetic and built here, except ``fixtures/retype/id2label.json``: the golden class order. The API
reads a model's ``id2label`` by index, so changing that file is a serving-contract change."""
import doctest
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

import data_generation as dg
import evaluate
import retype as rt
import split_data as sd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CITY, PRIVATE = "toy-city", "hidden-town"


def test_doctests():
    failed, _ = doctest.testmod(rt)
    assert failed == 0


def test_class_order_is_the_golden_contract():
    with open(os.path.join(HERE, "fixtures", "retype", "id2label.json")) as f:
        golden = {int(k): v for k, v in json.load(f).items()}
    assert rt.ID2LABEL == golden
    assert rt.LABEL2ID["Other"] == 7 and rt.OTHER == "Other"
    assert rt.VALIDATOR_TYPES == ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
    assert rt.canonical_type("Occlusion") == "Other"


# ---------------------------------------------------------------------------------------------------------------------
# Type edits
# ---------------------------------------------------------------------------------------------------------------------

def edits_frame(rows):
    cols = ["label_edit_id", "label_id", "label_type", "user_id", "old_label_type", "new_label_type", "old_severity",
            "new_severity", "old_tags", "new_tags", "source", "edit_time", "label_validation_id"]
    out = []
    for i, (lid, old, new, user, t) in enumerate(rows):
        out.append({"label_edit_id": i, "label_id": lid, "label_type": new, "user_id": user, "old_label_type": old,
                    "new_label_type": new, "old_severity": 2, "new_severity": 2, "old_tags": '["a", "b"]',
                    "new_tags": "[]", "source": "LabelEdit", "edit_time": t, "label_validation_id": ""})
    return pd.DataFrame(out, columns=cols)


def test_type_changes_drops_same_type_and_ai_and_keeps_the_latest():
    e = edits_frame([(1, "CurbRamp", "CurbRamp", "u", "2026-10-01T00:00:00-07:00"),     # tag edit only
                     (2, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00"),
                     (2, "NoCurbRamp", "Obstacle", "v", "2026-10-03T00:00:00-07:00"),    # later: wins
                     (3, "Obstacle", "Other", rt.AI_USER_ID, "2026-10-01T00:00:00-07:00"),
                     (10, "SurfaceProblem", "Obstacle", "u", "2026-09-01T00:00:00+00:00")])
    ch = rt.type_changes(e)
    assert ch.label_id.tolist() == ["2", "10"]   # numeric order, ids as strings
    assert ch.set_index("label_id").loc["2", ["old_label_type", "new_label_type"]].tolist() == ["NoCurbRamp",
                                                                                                 "Obstacle"]


def test_type_changes_latest_wins_regardless_of_row_order():
    e = edits_frame([(2, "NoCurbRamp", "Obstacle", "v", "2026-10-03T00:00:00-07:00"),
                     (2, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00")])
    assert rt.type_changes(e).new_label_type.tolist() == ["Obstacle"]


@pytest.mark.parametrize("bad, match", [("seattle-wa:12", "mismatched city key forms"), ("12a", "per-city integer")])
def test_type_changes_refuses_other_key_forms(bad, match):
    e = edits_frame([(bad, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00")])
    with pytest.raises(ValueError, match=match):
        rt.type_changes(e)


def raw_label(lid, label_type, validations="[]", city=CITY, user="u", pano=None, lat=47.0, lon=-122.0, **kw):
    row = {"label_id": lid, "user_id": user, "pano_id": pano or f"p{lid}", "pano_source": "gsv",
           "label_type": label_type, "severity": 2, "tags": "[]", "validations": validations, "pano_x": 4000,
           "pano_y": 2000, "pano_width": 8192, "pano_height": 4096, "camera_pitch": 1.0, "camera_roll": 0.5,
           "latitude": lat, "longitude": lon, "city": city}
    row.update(kw)
    return row


def votes(agree, disagree):
    return json.dumps([{"user_id": f"a{i}", "validation": "Agree", "validator_type": "Human"} for i in range(agree)]
                      + [{"user_id": f"d{i}", "validation": "Disagree", "validator_type": "Human"}
                         for i in range(disagree)])


def test_join_current_labels_counts_every_drop():
    ch = rt.attach_city(pd.DataFrame({"label_id": ["1", "2", "3", "4", "5"],
                                      "old_label_type": ["CurbRamp", "Obstacle", "CurbRamp", "Occlusion", "Signal"],
                                      "new_label_type": ["NoCurbRamp", "Other", "Obstacle", "Other", "Problem"]}),
                        CITY)
    raw = pd.DataFrame([raw_label(1, "NoCurbRamp"), raw_label(2, "Obstacle"), raw_label(3, "Obstacle", pano_y=-5),
                        raw_label(4, "Other"), raw_label(5, "Problem"), raw_label(99, "CurbRamp")])
    kept, why = rt.join_current_labels(ch, raw)
    assert kept.label_uid.tolist() == [f"{CITY}:1"]
    assert why == {"type_mismatch": 1, "pano_y_out_of_range": 1, "same_class_after_alias": 1, "unknown_type": 1}
    ch2 = rt.attach_city(pd.DataFrame({"label_id": ["7"], "old_label_type": ["CurbRamp"],
                                       "new_label_type": ["NoCurbRamp"]}), CITY)
    assert rt.join_current_labels(ch2, raw)[1] == {"not_found": 1}


def test_join_is_on_the_composite_key_not_label_id():
    ch = rt.attach_city(pd.DataFrame({"label_id": ["1"], "old_label_type": ["CurbRamp"],
                                      "new_label_type": ["NoCurbRamp"]}), CITY)
    raw = pd.DataFrame([raw_label(1, "NoCurbRamp", city="elsewhere")])   # same label_id, another city
    kept, why = rt.join_current_labels(ch, raw)
    assert kept.empty and why == {"not_found": 1}


# ---------------------------------------------------------------------------------------------------------------------
# A synthetic round-3 data root
# ---------------------------------------------------------------------------------------------------------------------

def split_row(t, lid, cls, split, pano=None, city=CITY, severity="2", tags="[]"):
    return {"label_uid": f"{city}:{lid}", "city": city, "label_id": str(lid), "pano_id": pano or f"p{lid}",
            "label_type": t, "label_class": cls, "class_source": "margin", "author": "human", "human_agree": "3",
            "human_disagree": "0", "severity": severity, "tags": tags, "pano_source": "gsv", "hard": "0",
            "crop_path": f"{cls}/{city}_{lid}.webp", "status": "ok", "split": split}


def write_type(root, t, rows, crops=True):
    d = os.path.join(root, t)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "split.csv"), index=False)
    if crops:
        for r in rows:
            p = os.path.join(d, r["crop_path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()


def small_root(tmp_path):
    """Three types; CurbRamp's test row 1 shares panorama 'shared' with an Obstacle train row and a NoCurbRamp val
    row; Obstacle row 21 and NoCurbRamp row 12 share panorama 'pv' (val wins over train)."""
    root = str(tmp_path / "data")
    write_type(root, "CurbRamp", [split_row("CurbRamp", 1, "correct", "test", "shared"),
                                  split_row("CurbRamp", 2, "incorrect", "train"),
                                  split_row("CurbRamp", 3, "correct", "test_heldout", city=PRIVATE)])
    write_type(root, "NoCurbRamp", [split_row("NoCurbRamp", 11, "correct", "val", "shared"),
                                    split_row("NoCurbRamp", 12, "incorrect", "val", "pv")])
    write_type(root, "Obstacle", [split_row("Obstacle", 20, "correct", "train", "shared"),
                                  split_row("Obstacle", 21, "incorrect", "train", "pv"),
                                  split_row("Obstacle", 22, "correct", "train")])
    for t in ("SurfaceProblem", "Crosswalk"):
        write_type(root, t, [split_row(t, 30 + len(t), "correct", "train")])
    return root


def test_inherit_splits_takes_the_highest_priority_split_per_panorama(tmp_path):
    root = small_root(tmp_path)
    paths = rt.type_dirs(root, ["CurbRamp", "NoCurbRamp", "Obstacle"])
    v = pd.concat([rt.validated_rows(rt.read_split(p), t) for t, p in paths.items()], ignore_index=True)
    with pytest.raises(AssertionError):
        sd.check_no_shared_panoramas(v.assign(split=v.round3_split))
    out, moves = rt.inherit_splits(v)
    s = out.set_index("label_uid").split
    assert s[f"{CITY}:1"] == "test" and s[f"{CITY}:11"] == "test" and s[f"{CITY}:20"] == "test"
    assert s[f"{CITY}:21"] == "val" and s[f"{CITY}:12"] == "val" and s[f"{CITY}:22"] == "train"
    sd.check_no_shared_panoramas(out)
    assert moves.to_dict("records") == [{"type": "NoCurbRamp", "from": "val", "to": "test", "n": 1},
                                        {"type": "Obstacle", "from": "train", "to": "test", "n": 1},
                                        {"type": "Obstacle", "from": "train", "to": "val", "n": 1}]
    # every round-3 test / test_heldout row keeps its split
    kept = out[out.round3_split.isin(["test", "test_heldout"])]
    assert (kept.split == kept.round3_split).all()


def test_inherit_splits_rejects_an_unknown_split():
    d = pd.DataFrame({"label_uid": ["a:1"], "pano_id": ["p"], "claimed_type": ["CurbRamp"], "split": ["holdout"]})
    with pytest.raises(ValueError, match="unknown split"):
        rt.inherit_splits(d)


def test_inherit_splits_groups_rows_without_a_panorama_by_label():
    d = pd.DataFrame({"label_uid": ["a:1", "a:2"], "pano_id": ["", ""], "claimed_type": ["CurbRamp"] * 2,
                      "split": ["test", "train"]})
    assert rt.inherit_splits(d)[0].split.tolist() == ["test", "train"]


def type_edit_kept(lid=500, old="CurbRamp", new="NoCurbRamp", pano="pe"):
    return pd.DataFrame([{**raw_label(lid, new, pano=pano), "label_id": str(lid), "label_uid": f"{CITY}:{lid}",
                          "old_label_type": old, "new_label_type": new, "edit_time": "2026-10-01",
                          "source": "LabelEdit"}])


def test_assemble_end_to_end(tmp_path):
    root = small_root(tmp_path)
    crop = tmp_path / "edit" / "crops" / "x.webp"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"")
    edits = pd.concat([type_edit_kept(500), type_edit_kept(1, pano="shared")], ignore_index=True)  # 1: in round 3
    crops = pd.DataFrame({"label_uid": [f"{CITY}:500", f"{CITY}:1"], "crop_path": [str(crop)] * 2,
                          "status": ["ok"] * 2})
    df, rec = rt.assemble(root, list(rt.VALIDATOR_TYPES) + ["Signal"], edits, crops, crop_root=root)
    assert list(df.columns) == rt.SCHEMA
    assert df.row_uid.is_unique
    te = df[df.source == "type_edit"]
    assert sorted(te.row_uid) == [f"{CITY}:500@CurbRamp", f"{CITY}:500@NoCurbRamp"]
    assert (te.split == rt.TEST_RETYPE).all() and set(te.target_type) == {"NoCurbRamp"}
    v = df[df.source == "validated"]
    assert ((v.target_type == "") == (v.is_correct.astype(int) == 0)).all()
    assert rec["type_edit"]["in_round3_dropped"] == 1 and rec["missing_types"] == ["Signal"]
    assert rec["missing_crops"] == 0
    assert set(df[df.claimed_type == "CurbRamp"].crop_path) >= {"CurbRamp/correct/toy-city_1.webp"}


def test_assemble_refuses_missing_crops_and_missing_validator_types(tmp_path):
    root = small_root(tmp_path)
    os.remove(os.path.join(root, "Obstacle", "correct", f"{CITY}_22.webp"))
    with pytest.raises(FileNotFoundError, match="crops do not exist"):
        rt.assemble(root, rt.VALIDATOR_TYPES, crop_root=root)
    df, rec = rt.assemble(root, rt.VALIDATOR_TYPES, crop_root=root, allow_missing_crops=True)
    assert rec["missing_crops"] == 1
    with pytest.raises(FileNotFoundError, match="five round-3 types"):
        rt.type_dirs(str(tmp_path / "empty"), ["CurbRamp"])
    assert rt.type_dirs(root, ["CurbRamp", "Signal"]) == {"CurbRamp": os.path.join(root, "CurbRamp", "split.csv")}


def test_assemble_leaves_out_an_extra_type_with_too_few_incorrect_rows(tmp_path):
    root = small_root(tmp_path)
    write_type(root, "Signal", [split_row("Signal", 600 + i, "incorrect" if i < 2 else "correct", "train")
                                for i in range(8)])
    write_type(root, "NoSidewalk", [split_row("NoSidewalk", 700 + i, "incorrect" if i < 3 else "correct", "train")
                                    for i in range(8)])
    df, rec = rt.assemble(root, list(rt.VALIDATOR_TYPES) + ["Signal", "NoSidewalk"], crop_root=root,
                          min_extra_incorrect=3)
    assert rec["too_few_incorrect"] == {"Signal": 2} and "Signal" not in rec["types"]
    assert set(df.claimed_type) >= {"NoSidewalk"} and "Signal" not in set(df.claimed_type)


def test_assemble_flags_colocated_proxies(tmp_path):
    root = small_root(tmp_path)
    m = 1 / 111195  # about one metre of latitude
    raw = pd.DataFrame([raw_label(2, "CurbRamp", votes(0, 3), user="u1", lat=47.0),          # incorrect CurbRamp
                        raw_label(900, "NoCurbRamp", votes(4, 0), user="u2", lat=47.0 + m),   # correct, 1 m away
                        raw_label(12, "NoCurbRamp", votes(0, 3), user="u1", lat=48.0),
                        raw_label(21, "Obstacle", votes(0, 3), user="u1", lat=49.0),
                        raw_label(1, "CurbRamp", votes(4, 0), lat=50.0),
                        raw_label(3, "CurbRamp", votes(4, 0), city=PRIVATE),
                        raw_label(11, "NoCurbRamp", votes(4, 0), lat=51.0),
                        raw_label(20, "Obstacle", votes(4, 0), lat=52.0),
                        raw_label(22, "Obstacle", votes(4, 0), lat=53.0)])
    raw = rt.with_label_uid(raw)
    df, rec = rt.assemble(root, rt.VALIDATOR_TYPES, raw=raw, crop_root=root)
    proxy = df.set_index("label_uid").proxy_type
    assert proxy[f"{CITY}:2"] == "NoCurbRamp" and proxy[f"{CITY}:12"] == "" and rec["colocated"]["n"] == 1


def test_assemble_flags_a_proxy_whose_partner_sits_on_a_train_panorama(tmp_path):
    """Label 12 (incorrect NoCurbRamp, val) gets its CurbRamp proxy from label 901, which sits on panorama p22: a
    train panorama (Obstacle 22). Label 2's partner 900 is on a panorama outside the split: no leak."""
    root = small_root(tmp_path)
    m = 1 / 111195
    raw = pd.DataFrame([raw_label(2, "CurbRamp", votes(0, 3), user="u1", lat=47.0),
                        raw_label(900, "NoCurbRamp", votes(4, 0), user="u2", lat=47.0 + m),
                        raw_label(12, "NoCurbRamp", votes(0, 3), user="u1", lat=48.0),
                        raw_label(901, "CurbRamp", votes(4, 0), user="u2", lat=48.0 + m, pano="p22"),
                        raw_label(21, "Obstacle", votes(0, 3), user="u1", lat=49.0)])
    df, rec = rt.assemble(root, rt.VALIDATOR_TYPES, raw=rt.with_label_uid(raw), crop_root=root)
    d = df.set_index("label_uid")
    assert d.loc[f"{CITY}:12", "proxy_type"] == "CurbRamp" and d.loc[f"{CITY}:12", "proxy_b_in_train"] == "1"
    assert d.loc[f"{CITY}:2", "proxy_type"] == "NoCurbRamp" and d.loc[f"{CITY}:2", "proxy_b_in_train"] == "0"
    assert (df.loc[df.proxy_type == "", "proxy_b_in_train"] == "").all()
    assert rec["colocated"]["b_in_train_by_split"] == {"val": 1}
    # without pano_id in rawLabels the flag stays empty rather than claiming "no leak"
    df2, _ = rt.assemble(root, rt.VALIDATOR_TYPES, raw=rt.with_label_uid(raw.drop(columns="pano_id")),
                         crop_root=root)
    assert (df2.proxy_b_in_train == "").all()


def test_colocated_eval_breaks_down_by_partner_in_train():
    pred = pd.DataFrame({"source": ["validated"] * 4 + ["type_edit"], "is_correct": [0, 0, 0, 1, 0],
                         "split": ["test", "test", "test_heldout", "test", "test_retype"],
                         "proxy_type": ["CurbRamp", "Obstacle", "CurbRamp", "", ""],
                         "proxy_b_in_train": ["1", "0", "0", "", ""],
                         "suggested_type": ["CurbRamp", "Signal", "CurbRamp", "x", "y"],
                         "suggested_top3": ["CurbRamp|a|b", "Signal|Obstacle|a", "CurbRamp|a|b", "x|y|z", "y|z|a"]})
    out = rt.colocated_eval(pred)
    assert out[["split", "proxy_b_in_train", "n"]].values.tolist() == [
        ["test", "all", 2], ["test", "0", 1], ["test", "1", 1], ["test_heldout", "all", 1], ["test_heldout", "0", 1]]
    assert out.top1_acc.tolist() == [0.5, 0.0, 1.0, 1.0, 1.0]
    # predictions written before the flag existed still report (one "" group)
    assert rt.colocated_eval(pred.drop(columns="proxy_b_in_train")).n.sum() == 6


def test_assemble_merge_radius_pulls_a_nearby_panorama_into_the_test_split(tmp_path):
    """Obstacle 22 (train, panorama p22) is 1 m from CurbRamp 1 (test, panorama 'shared'). Off by default; with
    --merge-radius-m 3 the two panoramas are one group and 22 moves to test. Labels far apart never merge."""
    root = small_root(tmp_path)
    m = 1 / 111195
    lats = {1: 47.0, 22: 47.0 + m, 2: 47.5, 3: 48.0, 11: 47.0, 12: 49.0, 20: 47.0, 21: 49.0,
            30 + len("SurfaceProblem"): 50.0, 30 + len("Crosswalk"): 51.0}
    raw = rt.with_label_uid(pd.DataFrame([raw_label(lid, "CurbRamp", lat=lat, city=PRIVATE if lid == 3 else CITY)
                                          for lid, lat in lats.items()]))
    off, _ = rt.assemble(root, rt.VALIDATOR_TYPES, raw=raw, crop_root=root)
    on, rec = rt.assemble(root, rt.VALIDATOR_TYPES, raw=raw, crop_root=root, merge_radius_m=3.0)
    assert off.set_index("label_uid").split[f"{CITY}:22"] == "train"
    s = on.set_index("label_uid").split
    assert s[f"{CITY}:22"] == "test" and s[f"{CITY}:2"] == "train" and rec["merge_radius_m"] == 3.0
    assert {"type": "Obstacle", "from": "train", "to": "test", "n": 2} in rec["moves"]  # 20 (pano) and 22 (radius)
    with pytest.raises(ValueError, match="rawlabels"):
        rt.assemble(root, rt.VALIDATOR_TYPES, crop_root=root, merge_radius_m=3.0)


def test_cmd_assemble_writes_split_and_record(tmp_path):
    root = small_root(tmp_path)
    out = tmp_path / "out" / "retype_split.csv"
    rt.main(["assemble", "--splits", root, "--types", ",".join(rt.VALIDATOR_TYPES), "--out", str(out)])
    rec = json.loads((tmp_path / "out" / "assemble.json").read_text())
    assert rec["sha256"] == rt.file_sha256(str(out)) and set(rec["types"]) == set(rt.VALIDATOR_TYPES)
    assert pd.read_csv(out, dtype=str, keep_default_na=False).columns.tolist() == rt.SCHEMA


# ---------------------------------------------------------------------------------------------------------------------
# Colocation
# ---------------------------------------------------------------------------------------------------------------------

M = 1 / 111195


def inc(lat=47.0, claimed="NoCurbRamp", user="u"):
    return pd.DataFrame({"label_uid": [f"{CITY}:1"], "city": [CITY], "claimed_type": [claimed], "user_id": [user],
                         "latitude": [lat], "longitude": [-122.0]})


def cor(*rows):
    return pd.DataFrame([{"city": CITY, "label_type": t, "user_id": u, "latitude": 47.0 + d * M,
                          "longitude": -122.0} for t, u, d in rows])


def test_colocated_one_type_two_types_same_user_and_same_type():
    assert rt.colocated_targets(inc(), cor(("CurbRamp", "v", 1.0)))[f"{CITY}:1"] == "CurbRamp"
    assert rt.colocated_targets(inc(), cor(("CurbRamp", "v", 1.0), ("CurbRamp", "w", 0.5)))[f"{CITY}:1"] == "CurbRamp"
    assert rt.colocated_targets(inc(), cor(("CurbRamp", "v", 1.0), ("Obstacle", "v", 1.0)))[f"{CITY}:1"] == ""
    assert rt.colocated_targets(inc(), cor(("CurbRamp", "u", 1.0)))[f"{CITY}:1"] == ""       # same user
    assert rt.colocated_targets(inc(), cor(("NoCurbRamp", "v", 1.0)))[f"{CITY}:1"] == ""     # same type
    assert rt.colocated_targets(inc(), cor(("CurbRamp", "v", 2.0)))[f"{CITY}:1"] == ""       # beyond 1.5 m
    assert rt.colocated_targets(inc(lat=float("nan")), cor(("CurbRamp", "v", 0.0)))[f"{CITY}:1"] == ""


def test_colocation_radius_is_split_data_haversine():
    c = cor(("CurbRamp", "v", 1.49))
    d = float(sd.haversine_m(47.0, -122.0, c.latitude[0], c.longitude[0]))
    assert rt.colocated_targets(inc(), c, radius_m=d)[f"{CITY}:1"] == "CurbRamp"
    assert rt.colocated_targets(inc(), c, radius_m=d * 0.999)[f"{CITY}:1"] == ""


def test_colocation_never_crosses_cities():
    c = cor(("CurbRamp", "v", 0.0)).assign(city="elsewhere")
    assert rt.colocated_targets(inc(), c)[f"{CITY}:1"] == ""


# ---------------------------------------------------------------------------------------------------------------------
# Weights and caps
# ---------------------------------------------------------------------------------------------------------------------

def test_class_weights_arithmetic():
    counts = {"CurbRamp": 5700, "NoCurbRamp": 10000, "Obstacle": 9300, "Crosswalk": 1700, "Other": 30}
    w = rt.class_weights(counts)
    assert set(w) == set(rt.CLASSES) and w["NoSidewalk"] == 1.0
    mean_w = sum(w[c] * n for c, n in counts.items()) / sum(counts.values())
    assert mean_w == pytest.approx(1.0)
    raw = {c: min(max((np.mean(list(counts.values())) / n) ** 0.5, 0.5), 4.0) for c, n in counts.items()}
    assert w["Other"] / w["CurbRamp"] == pytest.approx(raw["Other"] / raw["CurbRamp"])
    assert raw["Other"] == 4.0   # clipped
    assert set(rt.class_weights(counts, power=0).values()) == {1.0}


def cap_frame(n, cls="CurbRamp", start=0):
    return pd.DataFrame({"row_uid": [f"c:{i}@{cls}" for i in range(start, start + n)], "split": ["train"] * n,
                         "is_correct": [1] * n, "target_type": [cls] * n})


def test_cap_per_class_is_seeded_and_stable_under_an_added_row():
    d = pd.concat([cap_frame(50), cap_frame(3, "Crosswalk", 100),
                   cap_frame(5).assign(split="val", row_uid=lambda x: x.row_uid + "v")], ignore_index=True)
    kept, dropped = rt.cap_per_class(d, 10, seed=42)
    assert dropped == {"CurbRamp": 40}
    assert (kept.split == "val").sum() == 5 and (kept.target_type == "Crosswalk").sum() == 3
    more = pd.concat([d, cap_frame(1, start=999)], ignore_index=True)
    kept2, _ = rt.cap_per_class(more, 10, seed=42)
    a = set(kept[kept.split == "train"].row_uid)
    b = set(kept2[kept2.split == "train"].row_uid)
    assert len(a - b) <= 1 and len(b - a) <= 1
    assert set(rt.cap_per_class(d, 10, seed=7)[0].row_uid) != set(kept.row_uid)
    assert rt.cap_per_class(d, None)[0] is d


# ---------------------------------------------------------------------------------------------------------------------
# Probabilities -> verdicts and suggestions
# ---------------------------------------------------------------------------------------------------------------------

PROBS = np.array([[0.60, 0.20, 0.05, 0.05, 0.02, 0.02, 0.02, 0.04],
                  [0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.60],
                  [0.05, 0.70, 0.05, 0.05, 0.05, 0.03, 0.03, 0.04]])


def test_p_claimed_and_suggestions_exclude_the_claimed_type():
    ci = rt.claimed_index(["CurbRamp", "CurbRamp", "NoCurbRamp"])
    assert rt.p_claimed_from_probs(PROBS, ci).tolist() == [0.60, 0.10, 0.70]
    assert [s[0] for s in rt.suggested_types(PROBS, ci)] == ["NoCurbRamp", "Other", "CurbRamp"]
    assert rt.suggested_types(PROBS, ci, 3)[0] == ["NoCurbRamp", "Obstacle", "SurfaceProblem"]
    with pytest.raises(ValueError, match="unknown claimed"):
        rt.claimed_index(["Driveway"])


def test_api_fields_other_and_correct_give_no_suggestion():
    p = dict(zip(rt.CLASSES, PROBS[1]))
    f = rt.api_fields(p, "CurbRamp")
    assert f["validation_result"] == "incorrect" and f["suggested_label_type"] is None
    assert f["validation_score"] == pytest.approx(0.9)
    f = rt.api_fields(dict(zip(rt.CLASSES, PROBS[0])), "CurbRamp")
    assert f["validation_result"] == "correct" and f["suggested_label_type"] is None
    f = rt.api_fields(dict(zip(rt.CLASSES, PROBS[2])), "CurbRamp")
    assert (f["suggested_label_type"], f["suggested_label_type_confidence"]) == ("NoCurbRamp", 0.70)
    with pytest.raises(ValueError):
        rt.api_fields({"CurbRamp": 1.0}, "CurbRamp")


def test_predictions_frame_columns():
    rows = pd.DataFrame({"row_uid": ["a:1@CurbRamp", "a:2@CurbRamp", "a:3@NoCurbRamp"], "label_uid": ["a:1", "a:2",
                         "a:3"], "split": ["test"] * 3, "claimed_type": ["CurbRamp", "CurbRamp", "NoCurbRamp"],
                         "target_type": ["CurbRamp", "", "NoCurbRamp"], "is_correct": [1, 0, 1],
                         "source": ["validated"] * 3, "proxy_type": ["", "Other", ""]})
    p = rt.predictions_frame(rows, PROBS)
    assert list(p.columns) == rt.PREDICTION_COLUMNS
    assert p.top1.tolist() == ["CurbRamp", "Other", "NoCurbRamp"]
    assert p.top3[0] == "CurbRamp|NoCurbRamp|Obstacle" and p.suggested_type[1] == "Other"


# ---------------------------------------------------------------------------------------------------------------------
# A scored synthetic run: export, evaluate.py, calibrate, the paired comparison, the report
# ---------------------------------------------------------------------------------------------------------------------

SPLITS = ["train"] * 30 + ["val"] * 30 + ["test"] * 30 + ["test_heldout"] * 20


def scored_root(tmp_path, seed=0):
    """Five types x 110 rows (train/val/test/test_heldout), round-3 predictions.csv, deployment priors, the
    assembled retype split and retype predictions in which the claimed-type probability tracks the true class."""
    rng = np.random.default_rng(seed)
    root = str(tmp_path / "data")
    priors = {}
    r3_cols = ["label_uid", "split", "true", "p_correct", "p_incorrect"]
    for k, t in enumerate(rt.VALIDATOR_TYPES):
        rows = []
        for i, s in enumerate(SPLITS):
            lid = 1000 * (k + 1) + i
            city = PRIVATE if s == "test_heldout" else CITY
            cls = "incorrect" if i % 3 == 0 else "correct"
            rows.append(split_row(t, lid, cls, s, city=city, severity=str(1 + i % 3),
                                  tags='["steep"]' if i % 2 else "[]"))
        write_type(root, t, rows)
        sp = pd.DataFrame(rows)
        ev = sp[sp.split != "train"]
        y = (ev.label_class == "incorrect").to_numpy()
        p_inc = np.clip(0.3 * y + rng.uniform(0, 0.7, len(ev)), 0, 1)
        pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                      "p_correct": 1 - p_inc, "p_incorrect": p_inc}, columns=r3_cols).to_csv(
            os.path.join(root, t, "predictions.csv"), index=False)
        priors[t] = {"correct": 900, "incorrect": 100, "measured_on": "2026-10-19"}
    with open(os.path.join(root, "deployment_priors.json"), "w") as f:
        json.dump(priors, f)
    split, _ = rt.assemble(root, rt.VALIDATOR_TYPES, crop_root=root)
    out = tmp_path / "out"
    out.mkdir()
    split_path = out / "retype_split.csv"
    split.to_csv(split_path, index=False)
    ev = split[split.split != "train"].reset_index(drop=True)
    probs = rng.dirichlet(np.ones(8), len(ev))
    ci = rt.claimed_index(ev.claimed_type)
    good = ev.is_correct.astype(int).to_numpy() == 1
    probs[np.arange(len(ev)), ci] += np.where(good, 1.0, 0.2) * rng.uniform(0.5, 1.5, len(ev))
    probs /= probs.sum(axis=1, keepdims=True)
    pred = rt.predictions_frame(ev, probs)
    pred_path = out / "retype_predictions.csv"
    pred.to_csv(pred_path, index=False)
    (tmp_path / "public.txt").write_text(CITY + "\n")
    return {"root": root, "split": str(split_path), "pred": str(pred_path), "public": str(tmp_path / "public.txt"),
            "split_df": rt.read_split(str(split_path)), "pred_df": rt.read_predictions(str(pred_path))}


@pytest.fixture
def scored(tmp_path):
    return scored_root(tmp_path)


def test_export_binary_excludes_moved_rows_and_feeds_evaluate(scored, tmp_path):
    pred, split = scored["pred_df"], scored["split_df"]
    moved = split[(split.split != split.round3_split) & (split.source == "validated")]
    counts = rt.export_binary(pred, split, scored["root"], "retype-test")
    b = pd.read_csv(os.path.join(scored["root"], "CurbRamp", "predictions_retype-test.csv"))
    assert list(b.columns) == evaluate.PREDICTION_COLUMNS
    assert not set(b.label_uid) & set(moved.label_uid)
    assert counts["CurbRamp"] == {"val": 30, "test": 30, "test_heldout": 20}
    assert np.allclose(b.p_correct + b.p_incorrect, 1)
    m = evaluate.report(b)
    assert m["test"]["n"] == 30 and m["test"]["auc"] is not None
    evaluate.main(["--label_type", "CurbRamp", "--data-dir", scored["root"], "--skip-inference", "--tag",
                   "retype-test", "--per-city", "--min-city-n", "5", "--slice-boot", "0"])
    assert os.path.exists(os.path.join(scored["root"], "CurbRamp", "metrics_retype-test.json"))
    from calibration import calibrate as cal
    pi, _ = cal.load_prior(os.path.join(scored["root"], "deployment_priors.json"), "CurbRamp")
    api_map, rep, _ = cal.calibrate(b, pi)
    assert set(api_map) == {"correct", "incorrect"} and rep["n_fit"] == 30


def test_export_binary_refuses_an_empty_tag(scored):
    with pytest.raises(ValueError, match="tag"):
        rt.export_binary(scored["pred_df"], scored["split_df"], scored["root"], "")


def test_export_binary_drops_rows_moved_into_val(tmp_path):
    split = pd.DataFrame({"row_uid": ["a:1@CurbRamp", "a:2@CurbRamp"], "round3_split": ["train", "val"]})
    pred = pd.DataFrame({"row_uid": ["a:1@CurbRamp", "a:2@CurbRamp"], "label_uid": ["a:1", "a:2"],
                         "split": ["val", "val"], "claimed_type": ["CurbRamp"] * 2, "is_correct": [1, 0],
                         "source": ["validated"] * 2, "p_claimed": [0.9, 0.2]})
    assert rt.binary_rows(pred, split, "CurbRamp").label_uid.tolist() == ["a:2"]


def binaries(scored):
    out, r3, meta = {}, {}, {}
    for t in rt.VALIDATOR_TYPES:
        out[t] = rt.binary_rows(scored["pred_df"], scored["split_df"], t)
        r3[t] = pd.read_csv(os.path.join(scored["root"], t, "predictions.csv"))
        meta[t] = rt.read_split(os.path.join(scored["root"], t, "split.csv"))
    return out, r3, meta


def test_paired_auc_table_on_identical_rows(scored):
    b, r3, meta = binaries(scored)
    t = rt.paired_auc_table(b, r3, meta, n_boot=50, seed=1)
    assert set(t.split) == {"test", "test_heldout"} and len(t) == 10
    assert (t.n == np.where(t.split == "test", 30, 20)).all()
    assert np.allclose(t.delta, t.auc_retype - t.auc_round3)
    ok = t.delta_lo.notna()
    assert ok.any() and (t.delta_lo[ok] <= t.delta_hi[ok]).all()
    same = rt.paired_auc_table({"CurbRamp": r3["CurbRamp"]}, r3, meta, n_boot=20, seed=1)
    assert np.allclose(same.delta, 0) and np.allclose(same.delta_lo, 0)


def test_paired_auc_table_refuses_different_rows_or_classes(scored):
    b, r3, meta = binaries(scored)
    cr = b["CurbRamp"]
    short = {"CurbRamp": cr.drop(index=cr.index[cr.split == "test"][0])}
    with pytest.raises(ValueError, match="different rows"):
        rt.paired_auc_table(short, r3, meta, n_boot=0)
    flipped = b["CurbRamp"].copy()
    i = flipped.index[flipped.split == "test"][0]
    flipped.loc[i, "true"] = "incorrect" if flipped.loc[i, "true"] == "correct" else "correct"
    with pytest.raises(ValueError, match="disagree on the true class"):
        rt.paired_auc_table({"CurbRamp": flipped}, r3, meta, n_boot=0)


def test_type_head_aggregates_carry_no_ids(scored):
    pred = scored["pred_df"]
    conf = rt.confusion_correct(pred)
    assert conf.shape == (8, 8) and conf.values.sum() == int(((pred.source == "validated") & (pred.is_correct == 1)
                                                              & pred.split.isin(["test", "test_heldout"])).sum())
    for frame in (conf.reset_index(), rt.type_head_table(pred), rt.suggested_on_incorrect(pred).reset_index()):
        assert not set(frame.columns) & set(rt.ID_COLUMNS)
    s = rt.suggested_on_incorrect(pred)
    assert np.diag(s.values).sum() == 0   # the suggestion is never the claimed type


def test_report_end_to_end(scored, tmp_path):
    docs = tmp_path / "docs-out"
    rt.main(["report", "--predictions", scored["pred"], "--split", scored["split"], "--root", scored["root"],
             "--docs-dir", str(docs), "--n-boot", "20", "--slice-boot", "0", "--public-cities", scored["public"],
             "--job-ids", "111,222"])
    names = set(os.listdir(docs))
    assert {"binary_vs_round3.csv", "binary_vs_round3.md", "slices_vs_round3.md", "gate_what_if.csv",
            "type_head.csv", "confusion_correct_test.csv", "suggested_on_incorrect.csv", "type_edit_eval.csv",
            "colocated_eval.csv", "provenance.json"} <= names
    gate = pd.read_csv(docs / "gate_what_if.csv")
    assert set(gate.model) == {"retype", "round3"} and set(gate.verdict) == {"correct", "incorrect"}
    prov = json.loads((docs / "provenance.json").read_text())
    assert prov["sha256"]["retype_split.csv"] == rt.file_sha256(scored["split"]) and prov["job_ids"] == "111,222"
    for name in names:
        assert PRIVATE not in (docs / name).read_text()
    assert rt.check_aggregates(str(docs), [PRIVATE]) == len(names)


def test_check_aggregates_refuses_ids_tokens_and_private_names(tmp_path):
    d = tmp_path / "agg"
    d.mkdir()
    (d / "ok.csv").write_text("type,n\nCurbRamp,3\n")
    assert rt.check_aggregates(str(d), [PRIVATE]) == 1
    (d / "bad.csv").write_text("label_uid,n\nx,1\n")
    with pytest.raises(ValueError, match="id column"):
        rt.check_aggregates(str(d))
    os.remove(d / "bad.csv")
    (d / "note.md").write_text(f"see {CITY}:123\n")
    with pytest.raises(ValueError, match="label-id-like"):
        rt.check_aggregates(str(d))
    (d / "note.md").write_text(f"pooled {PRIVATE} numbers\n")
    with pytest.raises(ValueError, match="private"):
        rt.check_aggregates(str(d), [PRIVATE])


def test_suggestion_eval_on_type_edit_rows():
    rows = pd.DataFrame({"is_correct": [1, 0], "target_type": ["NoCurbRamp"] * 2,
                         "suggested_type": ["CurbRamp", "NoCurbRamp"],
                         "suggested_top3": ["CurbRamp|Other|Signal", "NoCurbRamp|Other|Signal"]})
    assert rt.suggestion_eval(rows, "target_type").values.tolist() == [[1, 1.0, 1.0]]
    assert rt.suggestion_eval(rows.iloc[:1], "target_type").n.tolist() == [0]


# ---------------------------------------------------------------------------------------------------------------------
# Stages with stubbed I/O: fetch-edits, type-edits, audit
# ---------------------------------------------------------------------------------------------------------------------

LISTING = {"fetched_on": "2026-11-05", "cities": [
    {"city_id": CITY, "visibility": "public", "url": "https://sidewalk-toy.cs.washington.edu"},
    {"city_id": PRIVATE, "visibility": "private", "url": None}]}


def edits_csv(rows):
    return edits_frame(rows).to_csv(index=False)


def test_fetch_edits_writes_one_csv_per_public_deployment(tmp_path):
    asked = []

    def get_text(url):
        asked.append(url)
        return edits_csv([(1, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00")])
    args = rt.build_parser().parse_args(["fetch-edits", "--out", str(tmp_path / "edits")])
    rec = rt.cmd_fetch_edits(args, get_text=get_text, fetch_listing=lambda url: LISTING)
    assert asked == ["https://sidewalk-toy.cs.washington.edu/v3/api/labelEdits?filetype=csv"]
    assert rec["cities"][CITY]["rows"] == 1
    saved = json.loads((tmp_path / "edits" / "fetch.json").read_text())
    assert saved["cities"][CITY]["sha256"] == rt.file_sha256(str(tmp_path / "edits" / f"{CITY}.csv"))


def test_fetch_edits_resolves_a_bare_id_through_the_listing_even_with_a_cached_csv(tmp_path):
    """A rawLabels export <cache>/<id>.csv must not make the id its own subdomain: the host comes from the listing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{CITY}.csv").write_text("label_id\n1\n")
    (cache / dg.CITIES_CACHE_FILE).write_text(json.dumps(LISTING))
    asked = []

    def get_text(url):
        asked.append(url)
        return edits_csv([])
    args = rt.build_parser().parse_args(["fetch-edits", "--out", str(tmp_path / "edits"), "--cities", CITY,
                                         "--cities-cache", str(cache)])
    rt.cmd_fetch_edits(args, get_text=get_text, fetch_listing=lambda url: LISTING)
    assert asked == ["https://sidewalk-toy.cs.washington.edu/v3/api/labelEdits?filetype=csv"]


def test_cmd_type_edits_end_to_end(tmp_path):
    ed = tmp_path / "edits"
    ed.mkdir()
    (ed / f"{CITY}.csv").write_text(edits_csv([(1, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00"),
                                               (2, "Obstacle", "Obstacle", "u", "2026-10-01T00:00:00-07:00"),
                                               (3, "Obstacle", "SurfaceProblem", "u", "2026-10-01T00:00:00-07:00")]))
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([raw_label(1, "NoCurbRamp"), raw_label(2, "Obstacle"), raw_label(3, "Obstacle")]).drop(
        columns="city").to_csv(raw / f"{CITY}.csv", index=False)
    out = tmp_path / "w" / "type_edits.csv"
    kept, rec = rt.main(["type-edits", "--edits", str(ed), "--rawlabels", str(raw), "--out", str(out)])
    assert kept.label_uid.tolist() == [f"{CITY}:1"] and rec["dropped"] == {"type_mismatch": 1}
    assert json.loads((tmp_path / "w" / "type_edits.json").read_text())["pairs"] == {"CurbRamp->NoCurbRamp": 1}


def test_cmd_audit_pools_private_deployments(tmp_path):
    root = small_root(tmp_path)
    ed = tmp_path / "edits"
    ed.mkdir()
    for c in (CITY, PRIVATE):
        (ed / f"{c}.csv").write_text(edits_csv([(1, "CurbRamp", "NoCurbRamp", "u", "2026-10-01T00:00:00-07:00")]))
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [raw_label(i, t, votes(*v), user=f"u{i}") for i, t, v in
            [(1, "CurbRamp", (4, 0)), (2, "CurbRamp", (0, 3)), (11, "NoCurbRamp", (4, 0)), (12, "NoCurbRamp", (0, 2)),
             (20, "Obstacle", (4, 0)), (21, "Obstacle", (0, 3)), (22, "Obstacle", (4, 0)),
             (36, "SurfaceProblem", (4, 0)), (39, "Crosswalk", (4, 0)), (40, "Occlusion", (0, 3))]]
    pd.DataFrame(rows).drop(columns="city").to_csv(raw / f"{CITY}.csv", index=False)
    pd.DataFrame([raw_label(3, "CurbRamp", votes(4, 0), city=PRIVATE)]).drop(columns="city").to_csv(
        raw / f"{PRIVATE}.csv", index=False)
    (tmp_path / "public.txt").write_text(CITY + "\n")
    out = tmp_path / "audit" / "audit.json"
    a = rt.main(["audit", "--rawlabels", str(raw), "--splits", root, "--edits", str(ed), "--public-cities",
                 str(tmp_path / "public.txt"), "--out", str(out)])
    assert set(a["type_changes"]["per_deployment"]) == {CITY, rt.POOLED_PRIVATE}
    assert a["type_changes"]["total"]["type_changes"] == 2
    assert a["eligible"]["CurbRamp"] == {"correct": 2, "incorrect": 1}
    assert a["eligible"]["Other"] == {"correct": 0, "incorrect": 1}
    assert a["pano_sharing"]["CurbRamp"]["on_shared_pano"] == 1
    assert a["rawlabels_cities"] == {"public": 1, "private_pooled": 1}
    assert PRIVATE not in out.read_text()


# ---------------------------------------------------------------------------------------------------------------------
# data_generation.py --correct-proportion (the one hook) and the job files
# ---------------------------------------------------------------------------------------------------------------------

def test_correct_proportion_flag_reaches_the_selection(tmp_path, capsys):
    rows = [{**raw_label(i, "NoSidewalk", votes(4, 0) if i >= 3 else votes(0, 3)), "high_quality_user": True,
             "image_capture_date": "2024-01"} for i in range(30)]
    pd.DataFrame(rows).drop(columns="city").to_csv(tmp_path / f"{CITY}.csv", index=False)
    (tmp_path / dg.CITIES_CACHE_FILE).write_text(json.dumps(LISTING))
    base = ["--label_type", "NoSidewalk", "--rawlabels-cache", str(tmp_path), "--out", str(tmp_path / "out"),
            "--cities", CITY, "--select-only"]
    dg.main(base)
    assert "Correct Examples: 3" in capsys.readouterr().out          # 0.55 / 0.45 x 3 incorrect
    dg.main(base + ["--correct-proportion", "0.9"])
    assert "Correct Examples: 27" in capsys.readouterr().out         # all of them
    assert dg.build_parser().parse_args(["--label_type", "x"]).correct_proportion == dg.CORRECT_PROPORTION
    for bad in ("0", "1", "1.5", "-0.2"):
        with pytest.raises(SystemExit):
            dg.build_parser().parse_args(["--label_type", "x", "--correct-proportion", bad])


@pytest.mark.parametrize("name", ["retype.slurm", "retype_submit.sh"])
def test_job_files_parse(name):
    subprocess.run(["bash", "-n", os.path.join(REPO, name)], check=True)


def test_predict_on_a_tiny_checkpoint(tmp_path):
    """The GPU stage's loop on a CPU (torch only): a tiny random 8-class model scores every requested row."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    torch.manual_seed(0)
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32,
                                       patch_size=16, num_labels=8, id2label=rt.ID2LABEL, label2id=rt.LABEL2ID)
    ckpt = tmp_path / "ckpt"
    transformers.Dinov2ForImageClassification(config).save_pretrained(ckpt)
    transformers.BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(ckpt)
    root = tmp_path / "data"
    rows = []
    for i, (s, t, c) in enumerate([("train", "CurbRamp", 1), ("test", "CurbRamp", 0), ("val", "Obstacle", 1),
                                   ("test_retype", "NoCurbRamp", 0)]):
        p = root / f"{i}.png"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (40, 40), (10 * i, 20, 30)).save(p)
        rows.append({**{k: "" for k in rt.SCHEMA}, "row_uid": f"{CITY}:{i}@{t}", "label_uid": f"{CITY}:{i}",
                     "split": s, "claimed_type": t, "target_type": t if c else "", "is_correct": c,
                     "source": "validated", "crop_path": f"{i}.png"})
    split = tmp_path / "retype_split.csv"
    pd.DataFrame(rows, columns=rt.SCHEMA).to_csv(split, index=False)
    out = tmp_path / "pred" / "retype_predictions.csv"
    rt.main(["predict", "--run-dir", str(tmp_path), "--checkpoint", str(ckpt), "--split", str(split), "--crop-root",
             str(root), "--out", str(out), "--allow-cpu"])
    p = rt.read_predictions(str(out))
    assert list(p.columns) == rt.PREDICTION_COLUMNS and p.split.tolist() == ["test", "val", "test_retype"]
    assert np.allclose(p[rt.PROB_COLUMNS].sum(axis=1), 1, atol=1e-5)
    assert json.loads((tmp_path / "pred" / "retype_predictions.json").read_text())["rows"] == 3


def test_crops_stage_with_a_stubbed_store_and_depth_model(tmp_path, monkeypatch):
    """cmd_crops on CPU: the pose per provider reaches the renderer, a missing panorama becomes a status row, an
    error in one label does not end the run, consecutive labels on one panorama read it once, and a rerun skips the
    ok rows (resume on manifest.csv)."""
    labels = pd.DataFrame([{**raw_label(1, "NoCurbRamp", pano="pa"), "label_uid": f"{CITY}:1"},
                           {**raw_label(2, "Obstacle", pano="pa"), "label_uid": f"{CITY}:2"},
                           {**raw_label(3, "CurbRamp", pano="gone"), "label_uid": f"{CITY}:3"},
                           {**raw_label(4, "Other", pano="pb", pano_x="bad"), "label_uid": f"{CITY}:4"}])
    labels_csv = tmp_path / "type_edits.csv"
    labels.to_csv(labels_csv, index=False)
    reads, renders = [], []

    def read_store_panorama(store, city, pano_id):
        reads.append(pano_id)
        if pano_id == "gone":
            return "missing_pano", {}
        return np.zeros((16, 32, 3), np.uint8), {"pano_path": f"{city}/{pano_id}.jpg"}

    def create_perspective_crop(depth_model, pano_rgb, x, y, **kw):
        renders.append((round(x, 4), kw["apply_tilt"], kw["beta"], kw["camera_pitch"]))
        return np.zeros((8, 8, 3), np.uint8), {"depth_m": 5.0, "crop_half_px": 100, "clipped": 0}
    monkeypatch.setattr(dg, "check_pano_store_marker", lambda store: "test")
    monkeypatch.setattr(dg, "resolve_device", lambda allow_cpu: None)
    monkeypatch.setattr(dg, "load_depth_model", lambda encoder, device: object())
    monkeypatch.setattr(dg, "read_store_panorama", read_store_panorama)
    monkeypatch.setattr(dg, "create_perspective_crop", create_perspective_crop)
    out = tmp_path / "edit_crops"
    argv = ["crops", "--labels", str(labels_csv), "--pano-store", str(tmp_path), "--out", str(out), "--allow-cpu",
            "--tilt-beta", "0.95"]
    run = rt.main(argv)
    m = pd.read_csv(out / "manifest.csv", dtype=str, keep_default_na=False).set_index("label_uid")
    assert m.loc[f"{CITY}:1", "status"] == "ok" and m.loc[f"{CITY}:2", "status"] == "ok"
    assert m.loc[f"{CITY}:3", "status"] == "missing_pano"
    assert m.loc[f"{CITY}:4", "status"] == "error:ValueError"
    assert m.loc[f"{CITY}:1", "pose_source"] == "gsv_db" and float(m.loc[f"{CITY}:1", "tilt_beta"]) == 0.95
    assert os.path.isabs(m.loc[f"{CITY}:1", "crop_path"]) and os.path.exists(m.loc[f"{CITY}:1", "crop_path"])
    assert renders == [(round(4000 / 8192, 4), True, 0.95, 1.0)] * 2
    assert reads == ["gone", "pa", "pb"]  # sorted by city, pano: pa is read once for two labels
    assert run["status_counts_this_run"] == {"ok": 2, "missing_pano": 1, "error:ValueError": 1}
    assert json.loads((out / "crops_run.json").read_text())["labels_sha256"] == rt.file_sha256(str(labels_csv))
    reads.clear()
    rt.main(argv)  # resume: only the two failed labels are tried again
    assert reads == ["gone", "pb"]
    assert len(pd.read_csv(out / "manifest.csv")) == 4
