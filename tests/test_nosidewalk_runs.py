"""analysis/nosidewalk_runs.py on synthetic labels (issue #102: validate the run, not the label)."""
import doctest
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis import nosidewalk_gate as gate
from analysis import nosidewalk_round as nr
from analysis import nosidewalk_runs as runs
from tests.test_nosidewalk_gate import write_pilot_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1 = "00000000-0000-4000-8000-0000000000e1"
E2 = "00000000-0000-4000-8000-0000000000e2"
EXPERTS = f"one={E1},two={E2}"
AI = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"


def votes(*pairs):
    return json.dumps([{"user_id": u, "validation": v, "validator_type": "Human"} for u, v in pairs])


def label(city, i, user, edge, verdict=None, side="left", label_class="", validations=None, **kw):
    """One rawLabels-like NoSidewalk row; ``verdict`` becomes expert 1's own vote."""
    v = validations if validations is not None else (
        votes((E1, "Agree" if verdict == "correct" else "Disagree")) if verdict else "[]")
    row = {"label_id": str(i), "user_id": user, "label_type": "NoSidewalk", "street_edge_id": edge, "street_side": side,
           "validations": v, "time_created": f"2026-01-01T00:{i % 60:02d}:00Z", "city": city, "tags": "[]",
           "severity": None, "label_class": label_class, "pano_id": f"p{city}{i}", "pano_source": "gsv",
           "pano_x": 10.0, "pano_y": 5.0, "pano_width": 100.0, "pano_height": 50.0, "latitude": 47.0 + i * 1e-5,
           "longitude": -122.0, "correct": "", "agree_count": 0}
    row.update(kw)
    return row


def frame(rows):
    df = pd.DataFrame(rows)
    df["label_uid"] = df["city"] + ":" + df["label_id"]
    df["face_id"] = runs.nosidewalk_faces.face_id(df)
    return df


def test_doctests():
    assert doctest.testmod(runs).failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# regression tests (each fails on the wrong implementation named in its docstring)
# ---------------------------------------------------------------------------------------------------------------------

def test_face_copied_truth_is_not_agreement():
    """Reading the face-copied ``label_class`` would give agreement 1.0; each label's own expert vote gives 0."""
    raw = frame([label("a", 1, "u", 7, "correct", label_class="correct"),
                 label("a", 2, "u", 7, "incorrect", label_class="correct")])
    d, _ = runs.item1_labels(raw, runs.expert_order(EXPERTS))
    assert runs.pair_agreement(d) == 0.0
    assert runs.pair_agreement(d.assign(verdict=raw["label_class"].to_numpy())) == 1.0


def test_label_id_shared_across_cities(tmp_path):
    """The same label_id and street_edge_id in two cities are two labels on two faces, never one run."""
    for city in ("a", "b"):
        pd.DataFrame([label(city, 1, "u", 7, "correct"), label(city, 2, "u", 7, "incorrect" if city == "b"
                                                                 else "correct")]).drop(columns="city").to_csv(
            tmp_path / f"{city}.csv", index=False)
    raw, shas = runs.load_pull(str(tmp_path), ["NoSidewalk"], {"a", "b"})
    assert len(raw) == 4 and raw["label_uid"].is_unique and set(shas) == {"a.csv", "b.csv"}
    raw = runs.add_runs(raw)
    assert raw["run_id"].nunique() == 2 and len(raw) == 4
    d, _ = runs.item1_labels(raw, runs.expert_order(EXPERTS))
    assert len(runs.pair_table(d)) == 2       # one pair per city, none across
    assert runs.pair_agreement(d) == 0.5
    ns = raw.assign(user_key=raw["user_id"], expert_placed=0)
    f = runs.neighbour_features(ns, ns, ns.iloc[:0])
    assert len(f) == 4 and (f["n_ns_face"] == 2).all()


def test_no_votes_in_features():
    """Changing votes, classes or ``correct`` must leave the features byte-identical."""
    ns = frame([label("a", i, "u" if i < 3 else "v", 7, "correct") for i in range(1, 5)])
    ns = ns.assign(user_key=ns["user_id"], expert_placed=0)
    contra = ns.iloc[:1].assign(label_type="CurbRamp")
    f1 = runs.neighbour_features(ns, ns, contra).to_csv(index=False)
    ns2 = ns.assign(validations=votes((E1, "Disagree")), correct="f", agree_count=9, label_class="incorrect",
                    own_expert_verdict="incorrect")
    contra2 = contra.assign(validations="[]", correct="t", label_class="correct")
    assert runs.neighbour_features(ns2, ns2, contra2).to_csv(index=False) == f1


def synthetic_step_a(n_faces=40, per_face=3, seed=0):
    """A step (a) table as ``step_a_table`` returns it: faces in five folds, label-level truth on some rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_faces):
        city = "x" if f % 2 else "y"
        bad = f % 3 == 0
        for j in range(per_face):
            i = f * per_face + j
            y = "incorrect" if bad else "correct"
            rows.append({"label_uid": f"{city}:{i}", "city": city, "face_id": f"{city}:{f}:left",
                         "run_id": f"{city}:{f}:left:L{j % 2}", "pano_id": f"p{i}", "fold": f % 5,
                         "latitude": 47 + j * 1e-4, "longitude": -122.0, "scored": 1, "in_split": int(j < 2),
                         "true": y if j < 2 else "", "truth": y if j != 1 else "",
                         "truth_source": ("split_label" if j == 0 else "own_expert" if j == 2 else ""),
                         "logit": (-1.5 if bad else 2.0) + rng.normal(0, 1.2), "expert_placed": int(j == 2),
                         "trained_pano": 0, "tags": '["ends abruptly"]' if j else "[]", "severity": None})
    return pd.DataFrame(rows)


def test_folds_do_not_split_a_face():
    t = synthetic_step_a()
    t.loc[0, "fold"] = (t.loc[1, "fold"] + 1) % 5
    with pytest.raises(ValueError, match="two folds"):
        runs.step_a_pooling(t, {"correct": 0.8, "incorrect": 0.2}, n_boot=0)


def write_pull(directory, split, user_of=lambda i: f"00000000-0000-4000-8000-{i:012d}", experts_vote=True):
    """Per-city rawLabels CSVs that reproduce ``split``'s rows and faces, plus one extra label per face."""
    os.makedirs(directory, exist_ok=True)
    rows = []
    for _, r in split.iterrows():
        edge, side = r["face_id"].split(":")[1:]
        i = int(r["label_id"])
        v = votes((E1, "Agree" if r["label_class"] == "correct" else "Disagree")) if experts_vote else "[]"
        rows.append(label(r["city"], i, user_of(i % 4), int(edge), validations=v, side=side, pano_id=r["pano_id"]))
        rows.append(label(r["city"], 100000 + i, user_of(i % 4), int(edge), side=side))
    df = pd.DataFrame(rows)
    for city, g in df.groupby("city"):
        g.drop(columns="city").to_csv(os.path.join(directory, f"{city}.csv"), index=False)
    return df


def cities_json(path, public, private=()):
    data = {"status": "OK", "cities": [{"city_id": c, "visibility": "public"} for c in public]
            + [{"city_id": c, "visibility": "private"} for c in private]}
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


def make_gate(tmp_path):
    split = write_pilot_split(tmp_path, n_faces=30, per_face=2, heldout_faces=6)
    g = tmp_path / "gate"
    assert gate.main(["folds", "--split", split, "--out", str(g), "--folds", "5", "--seed", "42"]) == 0
    return pd.read_csv(split, dtype=str, keep_default_na=False), g


def test_private_city_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERTS", EXPERTS)
    pull = tmp_path / "pull"
    rows = [label(c, i, "u", i // 3, "correct" if i % 2 else "incorrect") for c in ("pubcity", "hiddencity")
            for i in range(30)]
    write = pd.DataFrame(rows)
    os.makedirs(pull)
    for c, g in write.groupby("city"):
        g.drop(columns="city").to_csv(pull / f"{c}.csv", index=False)
    cj = cities_json(tmp_path / "cities.json", ["pubcity"], ["hiddencity"])
    conf = tmp_path / "c.conf"
    conf.write_text('city-params {\n  db-schema {\n    pubcity = "sidewalk_pub"\n  }\n  landing-page-url {\n'
                    '    prod {\n      pubcity = "https://p.example.edu"\n    }\n  }\n}\n')
    out = tmp_path / "agreement.json"
    assert runs.main(["agreement", "--ns-pull", str(pull), "--cityparams", str(conf), "--cities-json", cj,
                      "--n-perm", "20", "--n-boot", "20", "--allow-unpinned", "--out-json", str(out)]) == 0
    text = out.read_text()
    assert "hiddencity" not in text and "pubcity" in text
    with pytest.raises(ValueError, match="non-public"):
        runs.check_public_text("a table naming hiddencity", {"pubcity"}, ["pubcity", "hiddencity"])
    assert runs.check_public_text("hiddencity-old is another id", {"pubcity"}, ["pubcity", "hiddencity"]) is None


def test_no_user_id_in_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERTS", EXPERTS)
    split, g = make_gate(tmp_path)
    write_pull(tmp_path / "pull", split)
    cj = cities_json(tmp_path / "cities.json", sorted(set(split["city"])))
    out = tmp_path / "step-a"
    assert runs.main(["step-a-neighbours", "--pull", str(tmp_path / "pull"), "--gate", str(g), "--cities-json", cj,
                      "--out", str(out)]) == 0
    faces = pd.read_csv(out / "step_a_faces.csv", dtype=str, keep_default_na=False)
    assert len(faces) == 2 * len(split[split["face_id"] != ""])
    assert set(faces["expert_set"]) <= {"", "1"}
    for name in runs.STEP_A_OUTPUTS:
        if name != "labels.csv":    # labels.csv is the crop job's input, with the pull's columns
            assert not runs.UUID_RE.search((out / name).read_text()), name
    times = tmp_path / "times.csv"
    pd.DataFrame({"schema": ["sidewalk_a"], "label_id": ["1"], "label_type": ["NoSidewalk"], "expert_idx": [1],
                  "validation_result": ["Agree"], "start_timestamp": ["2026-01-01 00:00:00+00"],
                  "end_timestamp": ["2026-01-01 00:00:01+00"], "source": ["Validate"], "mission_id": [3],
                  "label_author_is_expert": ["f"]}).to_csv(times, index=False)
    t = runs.vote_times(str(times), {"sidewalk_a": "a"}, "2026-02-01")
    assert not runs.UUID_RE.search(repr(t)) and list(t) == [("a", "1", 1)]


def test_faces_from_the_defining_pull(tmp_path):
    split, g = make_gate(tmp_path)
    pull = write_pull(tmp_path / "pull", split)
    sp, trained, _ = runs.gate_split(str(g))
    raw, _ = runs.load_pull(str(tmp_path / "pull"), ["NoSidewalk"], set(split["city"]))
    moved = raw.copy()
    first = sp.loc[sp["face_id"] != "", "label_uid"].iloc[0]
    moved.loc[moved["label_uid"] == first, "street_edge_id"] = 99999
    moved["face_id"] = runs.nosidewalk_faces.face_id(moved)
    with pytest.raises(ValueError, match="another face"):
        runs.step_a_faces(moved, sp, trained, runs.expert_order(EXPERTS))
    faces = runs.step_a_faces(raw, sp, trained, runs.expert_order(EXPERTS))
    assert set(faces.loc[faces["in_split"] == 1, "face_id"]) <= set(sp["face_id"])
    kept, counts = runs.face_check(raw[raw["label_uid"] == first], moved)
    assert kept == set() and counts == {"same_face": 0, "moved": 1, "missing": 0}
    assert len(pull) == 2 * len(split)


def test_pooled_score_needs_pooled_map():
    t = synthetic_step_a()
    pi = {"correct": 0.8, "incorrect": 0.2}
    fit = t[t["in_split"] == 1].assign(pooled=runs.pooled_scores(t, "face").loc[t["in_split"] == 1])
    pmap = runs.pooled_map(fit, pi, "face")
    assert pmap["pooled_group"] == "face"
    runs.serve_pooled(fit, pmap, "face")
    with pytest.raises(ValueError, match="needs a map fitted"):
        runs.serve_pooled(fit, pmap, "run")
    with pytest.raises(ValueError, match="untagged"):
        runs.serve_pooled(fit, pmap["map"], "face")


def test_step_b_join_on_city_uid():
    key = pd.DataFrame({"packet_id": ["p1", "p2"], "label_uid": ["sidewalk_a:1", "sidewalk_a:2"],
                        "label_uid_city": ["a:1", "a:2"], "city_id": "a", "weight": ["1.5", "2"],
                        "stratum": "pass"})
    fs = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "rated": [1, 1, 0], "face_id": ["a:1:left", "a:2:left",
                                                                                         "a:1:left"]})
    m = runs.join_key_scores(key, fs)
    assert len(m) == 2 and m["label_uid"].tolist() == ["a:1", "a:2"]
    with pytest.raises(ValueError, match="label_uid_city"):
        runs.join_key_scores(key.assign(label_uid_city=key["label_uid"]), fs)   # what a join on label_uid would do
    with pytest.raises(ValueError):
        runs.join_key_scores(key.drop(columns="label_uid_city"), fs)


def test_ai_and_expert_placed_excluded_from_item1():
    raw = frame([label("a", 1, E1, 7, "correct"), label("a", 2, AI, 7, "correct"), label("a", 3, "u", 7, "correct"),
                 label("a", 4, "u", 7, "incorrect")])
    d, counts = runs.item1_labels(raw, runs.expert_order(EXPERTS))
    assert d["label_uid"].tolist() == ["a:3", "a:4"]
    assert counts["expert_placed"] == 1 and counts["ai_placed"] == 1


def test_within_city_null():
    """Two cities with different base rates and no run effect: within the city null, a pooled null flags it."""
    rng = np.random.default_rng(1)
    rows = []
    for city, p in (("a", 0.95), ("b", 0.05)):
        for r in range(150):
            for j in range(3):
                rows.append({"city": city, "run_id": f"{city}:{r}", "labeler_idx": r % 20,
                             "verdict": "correct" if rng.random() < p else "incorrect"})
    d = pd.DataFrame(rows)
    obs = runs.pair_agreement(d)
    within = runs.null_summary(obs, runs.permutation_null(d, ["city"], 300, 0))
    pooled = runs.null_summary(obs, runs.permutation_null(d.assign(all="x"), ["all"], 300, 0))
    assert within["p"] > 0.05
    assert pooled["p"] < 0.01


def test_vote_gap_strata():
    d = pd.DataFrame({"run_id": "r", "city": "a", "label_id": ["1", "2", "3", "4"],
                      "verdict": ["correct"] * 4, "e1": ["Agree", "Agree", "Agree", ""], "e2": ["", "", "", "Agree"],
                      "tags": "[]", "severity": None, "expert_set": ["1", "1", "1", "2"]})
    t0 = pd.Timestamp("2026-01-01T00:00Z")
    t1, t2 = pd.Timestamp("2026-01-01T03:00Z"), pd.Timestamp("2026-01-04T00:00Z")
    times = {("a", "1", 1): (t0, "Agree", "Validate"), ("a", "2", 1): (t1, "Agree", "Validate"),
             ("a", "3", 1): (t2, "Disagree", "Validate")}  # a re-vote after the pull: the pull's vote has no time
    p = runs.pair_table(d, times)
    got = sorted(zip(p["gap"], p["source_pair"]))
    assert got.count(("different expert", "different expert")) == 3
    assert ("1 h - 1 day", "Validate") in got and got.count(("unknown", "unknown")) == 2
    assert set(d["expert_set"]) <= {"1", "2", "1+2"}
    assert not runs.UUID_RE.search(p.to_csv())


# ---------------------------------------------------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------------------------------------------------

def test_pair_agreement_and_icc_by_hand():
    d = pd.DataFrame({"run_id": ["a"] * 3 + ["b"] * 2 + ["c"], "city": "x",
                      "verdict": ["correct", "correct", "incorrect", "incorrect", "incorrect", "correct"]})
    assert runs.pair_agreement(d) == pytest.approx(2 / 4)
    assert runs.pair_agreement(d, weight="run") == pytest.approx((1 / 3 + 1) / 2)
    assert runs.homogeneous_share(d) == 0.5
    # ICC(1) by hand: groups a (2/3), b (0); grand 2/5; n0 = (5 - 13/5) / 1 = 2.4
    msb = 3 * (2 / 3 - 0.4) ** 2 + 2 * 0.4 ** 2
    msw = (2 * (1 / 3) ** 2 + (2 / 3) ** 2) / 3
    assert runs.icc_binary(d) == pytest.approx((msb - msw) / (msb + 1.4 * msw))


def test_permutation_null_matches_analytic():
    rng = np.random.default_rng(3)
    p = 0.7
    d = pd.DataFrame({"run_id": np.repeat(np.arange(2000), 2).astype(str), "city": "x", "labeler_idx": 0,
                      "verdict": np.where(rng.random(4000) < p, "correct", "incorrect")})
    share = (d["verdict"] == "correct").mean()
    null = runs.permutation_null(d, ["city"], 200, 0)
    assert null.mean() == pytest.approx(share ** 2 + (1 - share) ** 2, abs=0.01)


def test_pooled_scores_variants():
    d = pd.DataFrame({"face_id": ["f"] * 4 + ["g"], "run_id": ["r", "r", "r", "s", "t"],
                      "pano_id": ["p", "p", "q", "q", "z"], "logit": [0.0, 2.0, 4.0, 8.0, 1.0],
                      "latitude": 0.0, "longitude": 0.0})
    assert runs.pooled_scores(d, "run").tolist() == [2.0, 2.0, 2.0, 8.0, 1.0]
    assert runs.pooled_scores(d, "run_pano").tolist() == [2.5, 2.5, 2.5, 8.0, 1.0]   # (mean(0,2), 4) -> 2.5
    assert runs.pooled_scores(d, "face_loo").tolist() == [14 / 3, 4.0, 10 / 3, 2.0, 1.0]
    assert runs.pooled_scores(d, "single").tolist() == d["logit"].tolist()
    no_fourth = pd.Series([True, True, True, False, True])
    assert runs.pooled_scores(d, "face", members=no_fourth).tolist()[:4] == [2.0, 2.0, 2.0, 3.5]


def test_transition_counts_and_weights():
    decided = pd.DataFrame({"stratum": ["pass", "pass", "agree_below", "agree_below", "model_incorrect"],
                            "verdict": ["correct", "incorrect", "correct", "incorrect", "correct"],
                            "weight": [1.0, 2.0, 10.0, 20.0, 5.0], "city_id": "a"})
    single = (decided["stratum"] == "pass").to_numpy()
    pooled = np.array([True, False, True, True, False])
    b = runs.gate_block(decided, pooled, single, n_boot=0, seed=0)
    tr = pd.DataFrame(b["transition"])
    assert tr["n"].sum() == 5 and tr["weight"].sum() == 38.0
    row = tr[(tr["stratum"] == "agree_below") & tr["pooled_pass"] & (tr["verdict"] == "incorrect")]
    assert row["weight"].tolist() == [20.0]
    assert b["added_outside_pass"] == 2 and b["newly_admitted_incorrect"] == 1 and b["dropped_incorrect"] == 1
    assert b["raw"]["k"] == 2 and b["raw"]["n"] == 3
    assert b["coverage_weighted"] == pytest.approx(31 / 38)
    assert b["precision_weighted"]["weighted"] == pytest.approx(11 / 31)


def test_pin_mismatch_refuses():
    with pytest.raises(SystemExit, match="not the pinned"):
        runs.check_pin("key.csv", "0" * 64, allow_unpinned=False)
    assert runs.check_pin("key.csv", "0" * 64, allow_unpinned=True) == "0" * 64


def test_report_splices_only_its_own_markers():
    doc = ("x\n<!-- round-b-results:start -->\nB\n<!-- round-b-results:end -->\n<!-- gate-results:start -->\nG\n"
           "<!-- gate-results:end -->\n" + runs.START + "\nold\n" + runs.END + "\ny\n")
    out = nr.splice(doc, "new\n", start=runs.START, end=runs.END)
    assert "\nB\n" in out and "\nG\n" in out and "old" not in out and "new" in out
    page = open(os.path.join(ROOT, "docs", "nosidewalk-runs-2026-11.md")).read()
    assert page.count(runs.START) == 1 and page.count(runs.END) == 1


def test_step_a_pooling_smoke():
    t = synthetic_step_a()
    out = runs.step_a_pooling(t, {"correct": 0.8, "incorrect": 0.2}, n_boot=20, seed=0,
                              train=t[["label_uid", "tags", "severity"]])
    assert set(out["groups"]) == set(runs.GROUPS)
    face = out["groups"]["face"]["ii_all"]
    assert face["n"] == int((t["truth"] != "").sum()) and face["delta_vs_single"]["delta"] is not None
    assert out["counts"]["set_iii"] == t.loc[t["truth"] != "", "face_id"].nunique()
    assert "no_expert_placed" in out["sensitivity"]


def test_recommendation_rule():
    b = {"added_outside_pass": 10, "newly_admitted_incorrect": 2, "raw": {"rate": 0.99}, "raw_single": {"rate": 0.986}}
    assert not runs.recommendation(b)["study_further"]
    assert runs.recommendation({**b, "newly_admitted_incorrect": 1})["study_further"]
    assert not runs.recommendation({**b, "newly_admitted_incorrect": 0, "raw": {"rate": 0.97}})["study_further"]


def test_doc_points_at_runs_page():
    text = open(os.path.join(ROOT, "docs", "nosidewalk-2026-11.md")).read()
    assert "it is not answered here" not in text
    assert "nosidewalk-runs-2026-11.md" in text
