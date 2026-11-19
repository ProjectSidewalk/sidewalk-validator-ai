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
from tests.test_nosidewalk_gate import write_fold_predictions, write_pilot_split

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
                      "--out", str(out), "--allow-unpinned"]) == 0
    with pytest.raises(SystemExit, match="not the pinned"):   # the pins are checked without --allow-unpinned
        runs.main(["step-a-neighbours", "--pull", str(tmp_path / "pull"), "--gate", str(g), "--cities-json", cj,
                   "--out", str(out)])
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


# ---------------------------------------------------------------------------------------------------------------------
# review round 1: tests that fail on the leak-critical mutations of items 2 and 3, and the report path
# ---------------------------------------------------------------------------------------------------------------------

def step_a_inputs():
    """Four labels for ``step_a_table``: two out-of-fold rows on a fold-2 face (a margin-truth split row and a split
    row whose class was copied from its face but which carries its own expert vote), one non-split row with its own
    expert verdict, and one held-out row; five fold models that give every label a different score."""
    faces = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "face_id": ["a:7:left"] * 3 + ["a:8:left"],
                          "heldout": [0, 0, 0, 1], "fold_of_face": [2.0, 2.0, 2.0, np.nan], "labeler_idx": [0, 0, 1, 0],
                          "in_split": [1, 1, 0, 1], "own_expert_verdict": ["", "incorrect", "incorrect", ""],
                          **{f"trained_pano_{k}": 0 for k in range(5)}})
    p = {k: pd.DataFrame({"label_uid": faces["label_uid"], "p_correct": [0.1 + 0.15 * k + 0.01 * i for i in range(4)]})
         for k in range(5)}
    split = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:4"], "label_class": ["correct", "correct", "incorrect"],
                          "class_source": ["margin", "face", "expert"], "split": ["test", "test", "test_heldout"],
                          "face_id": ["a:7:left", "a:7:left", "a:8:left"]})
    # as gate2 writes them: a split row is a test row of its own fold only; every fold scores the held-out rows
    gate_preds = {k: pd.DataFrame({"label_uid": ["a:1", "a:2", "a:4"], "split": ["test", "test", "test_heldout"],
                                   "true": ["correct", "correct", "incorrect"],
                                   "p_correct": [0.3 + 0.1 * k, 0.4, 0.2 + 0.1 * k]}).iloc[0 if k == 2 else 2:]
                  .assign(p_incorrect=lambda x: 1 - x["p_correct"]) for k in range(5)}
    return faces, p, split, gate_preds


def test_step_a_table_scores_each_face_by_its_own_fold():
    """An out-of-fold row takes only its fold's model (the mean of all five would leak the in-fold models); a
    held-out row takes the five-model mean; gate2's held-out logit is the same mean."""
    faces, p, split, gate_preds = step_a_inputs()
    t = runs.step_a_table(faces, p, split, gate_preds).set_index("label_uid")
    assert t.at["a:1", "logit"] == pytest.approx(float(runs.logit(p[2].at[0, "p_correct"])))
    assert t.at["a:3", "logit"] == pytest.approx(float(runs.logit(p[2].at[2, "p_correct"])))
    assert t.at["a:4", "logit"] == pytest.approx(np.mean([runs.logit(p[k].at[3, "p_correct"]) for k in range(5)]))
    assert t.at["a:1", "logit_gate2"] == pytest.approx(float(runs.logit(0.3 + 0.1 * 2)))
    assert t.at["a:4", "logit_gate2"] == pytest.approx(np.mean([runs.logit(0.2 + 0.1 * k) for k in range(5)]))
    assert (t["run_id"] == ["a:7:left:NoSidewalk:L0", "a:7:left:NoSidewalk:L0", "a:7:left:NoSidewalk:L1",
                            "a:8:left:NoSidewalk:L0"]).all()


def test_face_copied_class_is_not_label_level_truth():
    """A split row whose class the face rule copied has no label-level truth, even when it carries its own expert
    vote; a margin row keeps its class; a non-split row takes its own expert verdict."""
    faces, p, split, gate_preds = step_a_inputs()
    t = runs.step_a_table(faces, p, split, gate_preds).set_index("label_uid")
    assert (t.at["a:1", "truth"], t.at["a:1", "truth_source"]) == ("correct", "split_label")
    assert (t.at["a:2", "truth"], t.at["a:2", "truth_source"]) == ("", "")
    assert t.at["a:2", "true"] == "correct"          # still in the circular set (i)
    assert (t.at["a:3", "truth"], t.at["a:3", "truth_source"]) == ("incorrect", "own_expert")


def test_crop_parity_covers_heldout_rows():
    faces, p, split, gate_preds = step_a_inputs()
    t = runs.step_a_table(faces, p, split, gate_preds)
    rng = np.random.default_rng(0)   # more test rows of other labels, so the gate2 map can be fitted
    y = np.array(["correct", "incorrect"] * 40)
    pc = np.clip(np.where(y == "correct", 0.85, 0.3) + rng.normal(0, 0.1, 80), 0.01, 0.99)
    extra = pd.DataFrame({"label_uid": [f"z:{i}" for i in range(80)], "split": "test", "true": y, "p_correct": pc,
                          "p_incorrect": 1 - pc})
    gate_preds = {**gate_preds, 0: pd.concat([gate_preds[0], extra], ignore_index=True)}
    cp = runs.crop_parity(t, gate_preds, {"correct": 0.8, "incorrect": 0.2},
                          split=pd.concat([split, split.iloc[:1].assign(label_uid="a:9", face_id="")]))
    assert cp["rows"] == 2 and cp["heldout"]["rows"] == 1 and cp["split_rows_without_face"] == 1


def test_lofo_map_never_sees_its_own_fold(monkeypatch):
    t = synthetic_step_a()
    seen = []
    real = runs.pooled_map

    def spy(fit_rows, pi, group):
        seen.append(set(fit_rows["fold"]))
        return real(fit_rows, pi, group)

    monkeypatch.setattr(runs, "pooled_map", spy)
    fit = t[t["in_split"] == 1].assign(pooled=t["logit"])
    runs.lofo_served(fit, t.assign(pooled=t["logit"]), {"correct": 0.8, "incorrect": 0.2}, "single")
    assert seen == [set(range(5)) - {k} for k in range(5)]


def synthetic_second_stage(n_faces=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_faces):
        bad = f % 3 == 0
        for j in range(2):
            rows.append({"label_uid": f"x:{f}:{j}", "face_id": f"x:{f}:left", "city": "x" if f % 2 else "y",
                         "fold": f % 5, "truth": "incorrect" if bad else "correct",
                         "image_logit": (-1.0 if bad else 1.5) + rng.normal(0, 1),
                         "tags": '["ends abruptly"]', "verdict": "incorrect" if bad else "correct", "weight": 1.0 + j,
                         "city_id": "x" if f % 2 else "y", "stratum": "pass",
                         **{c: rng.normal() for c in runs.FEATURES},
                         **{f"p_correct_{k}": float(runs.sigmoid((-1.0 if bad else 1.5) + rng.normal(0, 1)))
                            for k in range(5)}})
    return pd.DataFrame(rows)


def test_second_stage_holds_out_its_fold(monkeypatch):
    train = synthetic_second_stage()
    sizes = []
    real = runs.fit_second_stage

    def spy(X, y, w):
        sizes.append(len(X))
        return real(X, y, w)

    monkeypatch.setattr(runs, "fit_second_stage", spy)
    out = runs.second_stage(train, {"ii_all": set(train["label_uid"])}, n_boot=0, seed=0)
    want = [len(train) - int((train["fold"] == k).sum()) for k in range(5) for _ in runs.MODEL_COLUMNS]
    assert sizes == want
    assert out["tags_ii_all"]["ends abruptly"]["rows"] == len(train)
    bad = train.copy()
    bad.loc[0, "fold"] = (bad.loc[1, "fold"] + 1) % 5   # one face on both sides of a fold
    with pytest.raises(ValueError, match="both sides"):
        runs.second_stage(bad, {"ii_all": set(bad["label_uid"])}, n_boot=0, seed=0)


def step_b_frames():
    """Two faces: face f with a rated label and three scored neighbours of one run (pass, pass, below, below), face
    g with a rated label and two neighbours (pass, below, below)."""
    rows = []
    for face, strata in (("c:1:left", ["pass", "pass", "agree_below", "agree_below"]),
                         ("c:2:left", ["agree_below", "pass", "agree_below"])):
        for j, st in enumerate(strata):
            rows.append({"label_uid": f"{face}:{j}", "face_id": face, "labeler_idx": 0, "pano_id": f"{face}{j}",
                         "latitude": 47.0, "longitude": -122.0, "stratum": st, "rater_placed": int(j == 1),
                         "rater_voted": 0, **{f"p_correct_{k}": 0.95 if st == "pass" else 0.6 for k in range(5)}})
    fs = pd.DataFrame(rows)
    fs["run_id"] = runs.run_ids(fs["face_id"], "NoSidewalk", fs["labeler_idx"]).to_numpy()
    rated = fs[fs["label_uid"].str.endswith(":0")].assign(verdict="correct", weight=1.0, city_id="c")
    return rated.reset_index(drop=True), fs


def test_vote_pooling_needs_half_of_the_group():
    """Face 1's run: 2 of 4 pass, so the vote passes; face 2's run: 1 of 3, so it does not (a threshold of 0 would
    pass both). Strict also needs the label itself to pass."""
    rated, fs = step_b_frames()
    assert runs.vote_pooled(rated, fs, "run").tolist() == [True, False]
    assert runs.vote_pooled(rated, fs, "run", strict=True).tolist() == [True, False]
    no_second = fs["rater_placed"] == 0      # drop the second (passing) label of each face: face 1 falls to 1 of 3
    assert runs.vote_pooled(rated, fs, "run", members=no_second).tolist() == [False, False]
    assert runs.vote_pooled(rated, fs.assign(stratum="agree_below"), "run").tolist() == [False, False]


def test_step_b_members_filter_changes_the_pool():
    rated, fs = step_b_frames()
    base = runs.step_b_pooled(rated, fs, "run")
    filt = runs.step_b_pooled(rated, fs, "run", members=fs["rater_placed"] == 0)
    hi, lo = runs.logit(0.95), runs.logit(0.6)
    assert base["pooled_0"].tolist() == pytest.approx([(2 * hi + 2 * lo) / 4, (hi + 2 * lo) / 3])
    assert filt["pooled_0"].tolist() == pytest.approx([(hi + 2 * lo) / 3, lo])


def test_near50_without_coordinates_keeps_own_score():
    d = pd.DataFrame({"face_id": ["f", "f"], "run_id": "r", "pano_id": ["p", "q"], "logit": [1.0, 3.0],
                      "latitude": [np.nan, 47.0], "longitude": [np.nan, -122.0]})
    assert runs.pooled_scores(d, "near50").tolist() == [1.0, 3.0]


def test_vote_times_as_of_and_city(tmp_path):
    """A vote after the pull is not the pull's vote; the same label_id in two cities is two labels."""
    times = tmp_path / "times.csv"
    pd.DataFrame({"schema": ["sidewalk_a", "sidewalk_a", "sidewalk_a", "sidewalk_b", "sidewalk_b"],
                  "label_id": ["1", "2", "1", "1", "2"], "label_type": "NoSidewalk", "expert_idx": 1,
                  "validation_result": ["Agree", "Agree", "Disagree", "Agree", "Agree"],
                  "start_timestamp": "", "source": "Validate", "mission_id": 1, "label_author_is_expert": "f",
                  "end_timestamp": ["2026-01-01 00:00:00+00", "2026-01-01 00:30:00+00", "2026-03-01 00:00:00+00",
                                    "2026-01-01 00:00:00+00", "2026-01-03 00:00:00+00"]}).to_csv(times, index=False)
    t = runs.vote_times(str(times), {"sidewalk_a": "a", "sidewalk_b": "b"}, "2026-02-01")
    assert sorted(t) == [("a", "1", 1), ("a", "2", 1), ("b", "1", 1), ("b", "2", 1)]
    assert t[("a", "1", 1)][1] == "Agree"      # the 2026-03 Disagree is after the pull
    d = pd.DataFrame({"run_id": ["ra", "ra", "rb", "rb"], "city": ["a", "a", "b", "b"],
                      "label_id": ["1", "2", "1", "2"], "verdict": "correct", "e1": "Agree", "tags": "[]",
                      "severity": None})
    p = runs.pair_table(d, t).set_index("city")
    assert p.at["a", "gap"] == "< 1 h" and p.at["b", "gap"] == ">= 1 day"


def test_pins_are_checked_by_name_not_basename(tmp_path):
    cj = cities_json(tmp_path / "anything.json", ["a"])
    with pytest.raises(SystemExit, match="cities_json"):
        runs.check_cities(cj, allow_unpinned=False)


def item1_agreement(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERTS", EXPERTS)
    pull = tmp_path / "pull"
    os.makedirs(pull)
    rows = [label("pubcity", i, f"u{(i // 3) % 5}", i // 3, "correct" if i % 4 else "incorrect") for i in range(90)]
    pd.DataFrame(rows).drop(columns="city").to_csv(pull / "pubcity.csv", index=False)
    cj = cities_json(tmp_path / "cities.json", ["pubcity"], ["hiddencity"])
    conf = tmp_path / "c.conf"
    conf.write_text('city-params {\n  db-schema {\n    pubcity = "sidewalk_pub"\n  }\n  landing-page-url {\n'
                    '    prod {\n      pubcity = "https://p.example.edu"\n    }\n  }\n}\n')
    out = tmp_path / "agreement.json"
    assert runs.main(["agreement", "--ns-pull", str(pull), "--cityparams", str(conf), "--cities-json", cj,
                      "--n-perm", "20", "--n-boot", "20", "--allow-unpinned", "--out-json", str(out)]) == 0
    return out, cj


def test_report_end_to_end_and_refuses_private_names(tmp_path, monkeypatch):
    agreement, cj = item1_agreement(tmp_path, monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text(f"head\n{runs.START}\n{runs.END}\ntail\n")
    args = ["report", "--agreement", str(agreement), "--cities-json", cj, "--out-json", str(tmp_path / "r.json"),
            "--out-md", str(tmp_path / "r.md"), "--doc", str(doc), "--allow-unpinned"]
    assert runs.main(args) == 0
    md = (tmp_path / "r.md").read_text()
    assert "Item 1" in md and "pair-weighted" in md and "pubcity" in (tmp_path / "r.json").read_text()
    assert md in doc.read_text() and not runs.UUID_RE.search(md + (tmp_path / "r.json").read_text())
    a = json.loads(agreement.read_text())
    a["settings"]["note"] = "from hiddencity"
    agreement.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="non-public"):
        runs.main(args)
    with pytest.raises(SystemExit, match="refusing to replace"):   # agreement output is guarded
        out = tmp_path / "agreement.json"
        runs.main(["agreement", "--ns-pull", str(tmp_path / "pull"), "--cityparams", str(tmp_path / "c.conf"),
                   "--cities-json", cj, "--n-perm", "20", "--n-boot", "20", "--allow-unpinned", "--out-json", str(out)])


def test_step_b_and_features_markdown():
    """The markdown of items 2 and 3 renders from synthetic step (a) / step (b) results, with the refitted-single
    comparison next to the rule (review R3)."""
    pi = {"correct": 0.8, "incorrect": 0.2}
    t = synthetic_step_a()
    a = runs.step_a_pooling(t, pi, n_boot=0, seed=0, groups=("single", "run", "face"),
                            train=t[["label_uid", "tags", "severity"]])
    fit = t[t["in_split"] == 1]
    maps = {g: runs.pooled_map(fit.assign(pooled=runs.pooled_scores(t, g).loc[fit.index]), pi, g) for g in runs.GROUPS}
    rated, fs = step_b_frames()
    rated = pd.concat([rated] * 20, ignore_index=True)
    rated["label_uid"] = [f"{u}#{i}" for i, u in enumerate(rated["label_uid"])]   # distinct rows, same groups
    rated = rated.assign(verdict=np.where(np.arange(len(rated)) % 4 == 0, "incorrect", "correct"),
                         stratum=np.where(np.arange(len(rated)) % 2 == 0, "pass", "agree_below"),
                         median_p_correct=0.9, tags='["ends abruptly"]')
    fs2 = pd.concat([fs, rated[fs.columns.drop(["stratum"]).tolist() + ["stratum"]]], ignore_index=True)
    b = runs.step_b_pooling(rated, fs2, maps, pi, n_boot=0, seed=0, groups=("single", "run"))
    assert set(b["groups"]["run"]) >= {"score", "vote", "vote_strict", "tags", "tags_vote"}
    b["singleton_runs"] = 0
    run = b["groups"]["run"]["score"]
    b["sensitivity"] = {"no_rater_placed_or_voted": run, "no_trained_pano_or_pilot_heldout": run,
                        "without_touched_faces": run, "touched_faces_rated": 0, "per_city": {"c": run}}
    rec = runs.recommendation(run, refit_single=b["groups"]["single"]["score"])
    cp = {"rows": 1, "median_abs_delta_logit": 0.0, "p90_abs_delta_logit": 0.0, "max_abs_delta_logit": 0.0,
          "verdict_flips": 0, "pass_flips": 0, "heldout": {"rows": 0}, "split_rows_without_face": 13}
    p = {"step_b": b, "recommendation": rec, "step_a": a, "crop_parity": cp,
         "r3_coverage": runs.r3_coverage(pd.DataFrame({"city": ["c"], "scored": [1]}))}
    md = "\n".join(runs.step_b_markdown(p) + runs.step_a_markdown(p))
    assert "What pooling adds" in md and "13 split rows without a face" in md and "Per city" in md
    train = synthetic_second_stage()
    blk = {"step_a": runs.second_stage(train, {"ii_all": set(train["label_uid"])}, n_boot=0, seed=0),
           "step_b": runs.transfer(train, train, 5, n_boot=0, seed=0)}
    f = {"face_check": {"step_a": {}, "step_b": {}}, "variants": {"without_expert_placed": blk}}
    fmd = "\n".join(runs.features_markdown(f))
    assert "By tag, without expert placed" in fmd and "ends abruptly" in fmd


# ---------------------------------------------------------------------------------------------------------------------
# review round 2
# ---------------------------------------------------------------------------------------------------------------------

def write_scores(root, name, n_folds, seed):
    d = root / name
    os.makedirs(d)
    for k in range(n_folds):
        pd.DataFrame({"label_uid": [f"c{seed}:{i}" for i in range(3)],
                      "p_correct": [0.1 * (seed + k + i + 1) for i in range(3)]}).to_csv(
            d / f"predictions_fold{k}.csv", index=False)
    return str(d)


def test_r3_pins_match_by_content_not_directory_name(tmp_path, monkeypatch):
    """S2: score directories copied back under other names are still checked; nothing pinned, a changed file or an
    extra file stops the run."""
    a, b = write_scores(tmp_path, "a", 2, 0), write_scores(tmp_path, "b", 2, 5)
    _, shas = runs.read_predictions([a, b], 2)
    pins = {f"r3_{n.replace('/', '_').removesuffix('.csv')}": v for n, v in shas.items()}
    monkeypatch.setattr(runs, "PINS", {k: v for k, v in runs.PINS.items() if not k.startswith("r3_")})
    with pytest.raises(SystemExit, match="no R3 prediction is pinned"):
        runs.check_r3_pins(shas, allow_unpinned=False)
    monkeypatch.setattr(runs, "PINS", {**runs.PINS, **pins})
    assert runs.check_r3_pins(shas, allow_unpinned=False) == shas
    renamed = [write_scores(tmp_path, "klone-a", 2, 0), write_scores(tmp_path, "klone-b", 2, 5)]
    _, shas2 = runs.read_predictions(renamed, 2)
    assert runs.check_r3_pins(shas2, allow_unpinned=False) == shas2
    changed = [write_scores(tmp_path, "klone-a2", 2, 1), b]
    _, shas3 = runs.read_predictions(changed, 2)
    with pytest.raises(SystemExit, match="do not match"):
        runs.check_r3_pins(shas3, allow_unpinned=False)
    with pytest.raises(SystemExit, match="do not match"):
        runs.check_r3_pins({k: v for k, v in shas.items() if "fold1" not in k}, allow_unpinned=False)
    with pytest.raises(ValueError, match="share the name"):      # two directories named a would share one key
        runs.read_predictions([a, write_scores(tmp_path / "x", "a", 2, 7)], 2)


def test_r3_coverage_threshold_stops_and_is_printed():
    """S3: under the plan's 90 % the run stops; the page prints the threshold, the outcome and the cities that are
    not fully scored."""
    t = pd.DataFrame({"city": ["s"] * 8 + ["o"] * 2, "scored": [1] * 6 + [0, 0] + [1, 1]})
    cov = runs.r3_coverage(t)
    assert cov["threshold"] == 0.9 and not cov["meets_threshold"]
    assert cov["per_city"]["s"] == {"labels": 8, "scored": 6, "share": 0.75}
    with pytest.raises(SystemExit, match="under the plan's 90%"):
        runs.require_coverage(cov)
    ok = runs.r3_coverage(pd.DataFrame({"city": ["s"] * 19 + ["o"], "scored": [1] * 18 + [0, 1]}))
    assert ok["meets_threshold"] and runs.require_coverage(ok) is ok
    t = synthetic_step_a()
    pi = {"correct": 0.8, "incorrect": 0.2}
    a = runs.step_a_pooling(t, pi, n_boot=0, seed=0, groups=("single", "run"), train=t[["label_uid", "tags", "severity"]])
    cp = {"rows": 1, "median_abs_delta_logit": 0.0, "p90_abs_delta_logit": 0.0, "max_abs_delta_logit": 0.0,
          "verdict_flips": 0, "pass_flips": 0}
    md = "\n".join(runs.step_a_markdown({"step_a": a, "crop_parity": cp, "r3_coverage": ok}))
    assert "the plan's threshold is 90 %: met" in md and "s 18 of 19" in md and "o 1 of 1" not in md


def test_step_a_table_refuses_a_face_in_two_folds():
    """S5: the two-fold check in step_a_table itself (the same condition is checked downstream too)."""
    faces, p, split, gate_preds = step_a_inputs()
    faces.loc[1, "fold_of_face"] = 3.0
    with pytest.raises(ValueError, match="two folds"):
        runs.step_a_table(faces, p, split, gate_preds)


def test_heldout_sensitivity_drops_every_label_on_a_step_a_heldout_face():
    """S6: pilot_heldout marks only the held-out split rows; a neighbour on the same held-out face with
    pilot_heldout = 0 must leave the held-out sensitivity too."""
    rated, fs = step_b_frames()
    fs = fs.assign(trained_pano=0, pilot_heldout=0)
    fs.loc[fs["label_uid"] == "c:1:left:2", "pilot_heldout"] = 1
    mem = runs.step_b_members(fs, rated, step_a_heldout={"c:1:left:2", "c:1:left:3"})
    m = dict(zip(fs["label_uid"], mem["no_trained_pano_or_pilot_heldout"]))
    assert not m["c:1:left:2"] and not m["c:1:left:3"] and m["c:1:left:1"] and m["c:2:left:1"]


# ---------------------------------------------------------------------------------------------------------------------
# review round 3
# ---------------------------------------------------------------------------------------------------------------------

def step_a_files(tmp_path, monkeypatch, unscored=0.0):
    """Synthetic files for ``pooling``'s step (a) side: a gate run with predictions, ``step_a_faces.csv`` from
    ``step-a-neighbours`` and two R3 score directories that leave a share ``unscored`` of the labels unscored. The
    R3 files are pinned in ``PINS`` (only them), so the guards run as in the published run."""
    monkeypatch.setenv("EXPERTS", EXPERTS)
    split, g = make_gate(tmp_path)
    for k in range(5):   # gate2's split.csv carries class_source (margin / expert / face)
        path = g / f"fold{k}" / "split.csv"
        f = pd.read_csv(path, dtype=str, keep_default_na=False)
        f.assign(class_source=np.where(f.index % 2 == 0, "margin", "face")).to_csv(path, index=False)
    write_fold_predictions(str(g))
    write_pull(tmp_path / "pull", split)
    cj = cities_json(tmp_path / "cities.json", sorted(set(split["city"])))
    out = tmp_path / "step-a"
    assert runs.main(["step-a-neighbours", "--pull", str(tmp_path / "pull"), "--gate", str(g), "--cities-json", cj,
                      "--out", str(out), "--allow-unpinned"]) == 0
    uids = pd.read_csv(out / "step_a_faces.csv", dtype=str)["label_uid"]
    uids = uids.iloc[:int(round(len(uids) * (1 - unscored)))]
    rng = np.random.default_rng(0)
    dirs = []
    for name, part in (("a", uids.iloc[::2]), ("b", uids.iloc[1::2])):
        d = tmp_path / "scores" / name
        os.makedirs(d)
        for k in range(5):
            pd.DataFrame({"label_uid": part, "p_correct": rng.uniform(0.05, 0.95, len(part))}).to_csv(
                d / f"predictions_fold{k}.csv", index=False)
        dirs.append(str(d))
    _, shas = runs.read_predictions(dirs, 5)
    monkeypatch.setattr(runs, "PINS", {f"r3_{i}": v for i, v in enumerate(shas.values())})
    args = runs.build_parser().parse_args(["pooling", "--step-a", str(out), "--scores", *dirs, "--gate", str(g),
                                           "--key", "k", "--sheet", "s", "--face-scores", "f", "--priors", "p",
                                           "--out-json", str(tmp_path / "pooling.json")])
    return args, dirs


def test_load_step_a_runs_the_r3_pin_check(tmp_path, monkeypatch):
    """T5: the content check is called where the predictions are read: a changed R3 file stops the run."""
    args, dirs = step_a_files(tmp_path, monkeypatch)
    t, *_ = runs.load_step_a(args)
    assert t["scored"].all() and (t["fold"] < 0).any()
    path = os.path.join(dirs[1], "predictions_fold3.csv")
    p = pd.read_csv(path)
    p.loc[0, "p_correct"] = 0.5
    p.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="do not match"):
        runs.load_step_a(args)


def test_load_step_a_runs_the_coverage_gate(tmp_path, monkeypatch):
    """T5: with a fifth of the step (a) labels unscored (and the files pinned), the run stops on coverage."""
    args, _ = step_a_files(tmp_path, monkeypatch, unscored=0.2)
    with pytest.raises(SystemExit, match="under the plan's 90%"):
        runs.load_step_a(args)


def pooling_frames():
    """A step (a) table with held-out faces (fold -1, split rows with a class) and step (b) frames, as ``pooling``
    gets them from ``load_step_a`` / ``load_step_b``."""
    t = synthetic_step_a()
    held = t["face_id"].isin(sorted(set(t["face_id"]))[-6:])
    t.loc[held, "fold"] = -1
    split = t[["label_uid", "tags", "severity"]].assign(split=np.where(held, "test_heldout", "test"))
    rated, fs = step_b_frames()
    rated = pd.concat([rated] * 20, ignore_index=True)
    rated["label_uid"] = [f"{u}#{i}" for i, u in enumerate(rated["label_uid"])]
    rated = rated.assign(verdict=np.where(np.arange(len(rated)) % 4 == 0, "incorrect", "correct"),
                         stratum=np.where(np.arange(len(rated)) % 2 == 0, "pass", "agree_below"),
                         median_p_correct=0.9, tags='["ends abruptly"]')
    fs = pd.concat([fs, rated[fs.columns.drop(["stratum"]).tolist() + ["stratum"]]], ignore_index=True)
    fs = fs.assign(trained_pano=0, pilot_heldout=0)
    return t, split, rated, fs


def test_pooling_end_to_end_fits_maps_out_of_fold_and_masks_heldout_faces(tmp_path, monkeypatch):
    """T5: through ``pooling`` itself, every map is fitted on out-of-fold rows only (never a held-out split row), the
    label-level sensitivity map on split-label rows only, and the held-out sensitivity is given every label on a
    step (a) held-out face."""
    t, split, rated, fs = pooling_frames()
    monkeypatch.setattr(runs, "load_step_a", lambda args: (t, split, {}, {}, 5))
    monkeypatch.setattr(runs, "load_step_b", lambda args: (rated, fs, {}))
    monkeypatch.setattr(runs, "crop_parity", lambda *a, **k: {})
    fits, held = [], []
    real_map, real_members = runs.pooled_map, runs.step_b_members

    def map_spy(fit_rows, pi, group):
        fits.append((group, set(fit_rows["fold"]), set(fit_rows["truth_source"])))
        return real_map(fit_rows, pi, group)

    def members_spy(fs_, rated_, step_a_heldout=()):
        held.append(set(step_a_heldout))
        return real_members(fs_, rated_, step_a_heldout=step_a_heldout)

    monkeypatch.setattr(runs, "pooled_map", map_spy)
    monkeypatch.setattr(runs, "step_b_members", members_spy)
    with open(tmp_path / "priors.json", "w") as f:
        json.dump({"NoSidewalk": {"correct": 80, "incorrect": 20, "source": "test"}}, f)
    out = tmp_path / "pooling.json"
    assert runs.main(["pooling", "--step-a", "x", "--scores", "x", "--gate", "x", "--key", "k", "--sheet", "s",
                      "--face-scores", "f", "--priors", str(tmp_path / "priors.json"), "--out-json", str(out),
                      "--n-boot", "0", "--allow-unpinned"]) == 0
    assert fits and all(min(folds) >= 0 for _, folds, _ in fits)
    assert any(src == {"split_label"} for _, _, src in fits)          # the label-level map
    assert held == [set(t.loc[t["fold"] < 0, "label_uid"])]
    p = json.loads(out.read_text())
    sens = p["step_b"]["sensitivity"]
    assert set(sens["face"]) >= {"no_rater_placed_or_voted", "no_trained_pano_or_pilot_heldout"}
    assert set(sens["label_level_map"]) == {"run", "face"} and sens["neighbours_removed"]["face"]["rater_placed"] > 0
    assert "delta_vs_single" in p["step_b"]["groups"]["run"]["score"]
    trained = split[split["split"] != "test_heldout"]          # held-out rows are not in the training pool
    assert p["step_b"]["groups"]["run"]["tags"]["ends abruptly"]["train_n"] == int(
        trained["tags"].str.contains("ends abruptly").sum())


def test_features_stops_without_experts(monkeypatch):
    """T3: without EXPERTS the without-expert-placed variant would equal the other one, so features stops first."""
    monkeypatch.delenv("EXPERTS", raising=False)
    with pytest.raises(SystemExit, match="EXPERTS is empty"):
        runs.main(["features", "--step-a", "x", "--scores", "x", "--gate", "x", "--key", "k", "--sheet", "s",
                   "--face-scores", "f", "--ns-pull", "n", "--round-b-pull", "r", "--contra-pull", "c",
                   "--cities-json", "j", "--out-json", "o.json"])


# ---------------------------------------------------------------------------------------------------------------------
# final review (S2): the lines that decide the two rule outcomes
# ---------------------------------------------------------------------------------------------------------------------

def item3_features(with_ci, without_ci, without_plain_ci):
    """A minimal ``features.json`` for :func:`runs.features_markdown`: the step (b) within-city delta b - a of each
    variant with the given intervals, and the plain (not within-city) delta of the pre-registered variant."""
    def step_b(within_ci, plain_ci):
        models = {m: {"auc_weighted": 0.9, "within_city_auc_weighted": 0.9} for m in runs.MODEL_COLUMNS}
        return {"decided": 100, "models": models,
                "delta_b_image_features_within_city_auc_weighted": {"delta": 0.02, "ci95": within_ci},
                "delta_b_image_features_auc_weighted": {"delta": 0.02, "ci95": plain_ci}}
    return {"face_check": {"step_a": {}, "step_b": {}},
            "variants": {"with_expert_placed": {"step_a": {}, "step_b": step_b(with_ci, [0.01, 0.03])},
                         "without_expert_placed": {"step_a": {}, "step_b": step_b(without_ci, without_plain_ci)}}}


def test_item3_rule_reads_the_within_city_lower_bound_without_expert_placed():
    """The item 3 rule reads only the pre-registered variant (without expert placed), only its within-city delta, and
    only the lower bound. Here the other variant, the plain delta and the upper bound are all above 0, and the rule
    must still say the features do not add."""
    md = "\n".join(runs.features_markdown(item3_features([0.01, 0.05], [-0.01, 0.05], [0.01, 0.03])))
    assert "**they do not add**" in md
    md = "\n".join(runs.features_markdown(item3_features([-0.05, -0.01], [0.002, 0.05], [-0.03, -0.01])))
    assert "**they add**" in md


def test_pooling_recommends_from_the_run_score_block(tmp_path, monkeypatch):
    """Item 4's rule is fed the run group's score-pooling block of the main step (b) call and the single group's
    score block as the refitted comparison: not the face pool, not vote pooling, not a sensitivity."""
    t, split, rated, fs = pooling_frames()
    monkeypatch.setattr(runs, "load_step_a", lambda args: (t, split, {}, {}, 5))
    monkeypatch.setattr(runs, "load_step_b", lambda args: (rated, fs, {}))
    monkeypatch.setattr(runs, "crop_parity", lambda *a, **k: {})
    calls, results = [], []
    real_rec, real_pool = runs.recommendation, runs.step_b_pooling

    def rec_spy(block, **kw):
        calls.append((block, kw.get("refit_single")))
        return real_rec(block, **kw)

    def pool_spy(*a, **kw):
        results.append(real_pool(*a, **kw))
        return results[-1]

    monkeypatch.setattr(runs, "recommendation", rec_spy)
    monkeypatch.setattr(runs, "step_b_pooling", pool_spy)
    with open(tmp_path / "priors.json", "w") as f:
        json.dump({"NoSidewalk": {"correct": 80, "incorrect": 20, "source": "test"}}, f)
    assert runs.main(["pooling", "--step-a", "x", "--scores", "x", "--gate", "x", "--key", "k", "--sheet", "s",
                      "--face-scores", "f", "--priors", str(tmp_path / "priors.json"),
                      "--out-json", str(tmp_path / "pooling.json"), "--n-boot", "0", "--allow-unpinned"]) == 0
    assert len(calls) == 1
    main = results[0]["groups"]
    assert calls[0][0] is main["run"]["score"] and calls[0][1] is main["single"]["score"]


def test_auc_delta_vs_single_subtracts_the_single_score():
    """Both arms of the paired delta: ``col`` minus the single label's ``median_p_correct``, never a column minus
    itself."""
    n = 24
    y = np.arange(n) % 4 == 0
    dd = pd.DataFrame({"verdict": np.where(y, "incorrect", "correct"), "weight": 1.0 + np.arange(n) % 3,
                       "city_id": "c", "stratum": "pass",
                       "pooled": np.where(y, 0.2, 0.9),                                   # separates perfectly
                       "median_p_correct": np.where(np.arange(n) % 3 == 0, 0.3, 0.8)})    # does not
    out = runs.auc_delta_vs_single(dd, "pooled", n_boot=0, seed=0)
    want = nr.weighted_auc(dd, "pooled") - nr.weighted_auc(dd, "median_p_correct")
    assert want > 0.1
    assert out["auc_weighted"]["delta"] == pytest.approx(want)
    assert out["within_city_auc_weighted"]["delta"] == pytest.approx(want)   # one city


def test_score_pooling_needs_three_of_five_models(monkeypatch):
    """A label passes the score gate when at least 3 of the 5 fold models serve it at or above the gate: 1 of 5 or 2
    of 5 is not enough."""
    monkeypatch.setattr(runs, "serve_pooled", lambda r, pmap, g: pd.DataFrame(
        {"verdict": "correct", "served": runs.sigmoid(r["pooled"])}, index=r.index))
    hi, lo = 0.99, 0.5
    rows = []
    for i, n_hi in enumerate((1, 2, 3, 5)):
        rows.append({"label_uid": f"c:{i}", "face_id": f"c:{i}:left", "labeler_idx": 0, "pano_id": f"p{i}",
                     "stratum": "agree_below", "verdict": "correct" if i else "incorrect", "weight": 1.0,
                     "city_id": "c", "median_p_correct": 0.9, "tags": "[]",
                     **{f"p_correct_{k}": hi if k < n_hi else lo for k in range(5)}})
    rated = pd.DataFrame(rows)
    b = runs.step_b_pooling(rated, rated.copy(), {"single": None}, None, gate=0.92, n_boot=0, seed=0,
                            groups=("single",))
    blk = b["groups"]["single"]["score"]
    assert blk["raw"]["n"] == 2 and blk["added_outside_pass"] == 2       # the 3-of-5 and 5-of-5 labels only


def test_newly_admitted_incorrect_counts_only_incorrect_labels():
    """Two labels newly admitted, both correct: 0 newly admitted incorrect (a count of correct ones would be 2)."""
    decided = pd.DataFrame({"stratum": ["pass", "agree_below", "agree_below", "model_incorrect"],
                            "verdict": ["correct", "correct", "correct", "incorrect"], "weight": 1.0,
                            "city_id": "a"})
    single = (decided["stratum"] == "pass").to_numpy()
    b = runs.gate_block(decided, np.array([True, True, True, False]), single, n_boot=0, seed=0)
    assert b["newly_admitted"] == 2 and b["newly_admitted_incorrect"] == 0


def test_strict_vote_needs_the_label_itself_to_pass():
    """Face 1's run passes the vote (2 of 4); with the rated label itself outside the pass stratum, strict fails."""
    rated, fs = step_b_frames()
    rated.loc[0, "stratum"] = "agree_below"     # the label, not its neighbours (fs keeps its share at 2 of 4)
    assert runs.vote_pooled(rated, fs, "run").tolist() == [True, False]
    assert runs.vote_pooled(rated, fs, "run", strict=True).tolist() == [False, False]


def test_transfer_scores_are_p_correct():
    """``transfer`` turns the second stage's p(incorrect) into p(correct) before ``weighted_auc`` (which reads
    p(correct)); without the flip an informative image logit would rank backwards."""
    train = synthetic_second_stage()
    out = runs.transfer(train, train, 5, n_boot=0, seed=0)
    assert out["models"]["a_image"]["auc_weighted"] > 0.8
    assert out["models"]["b_image_features"]["auc_weighted"] > 0.8


def test_face_weights_sum_to_one_per_face():
    w = runs.face_weights(pd.DataFrame({"face_id": ["f", "g", "g", "g"]}))
    assert w.tolist() == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])


def test_exclude_experts_drops_expert_placed_contradicting_labels():
    ns = pd.DataFrame({"label_uid": ["a:1"], "face_id": "a:1:left", "user_key": ["u"], "latitude": 0.0,
                       "longitude": 0.0, "expert_placed": [0]})
    contra = pd.DataFrame({"face_id": "a:1:left", "user_key": ["u", "e"], "expert_placed": [0, 1]})
    assert runs.neighbour_features(ns, ns, contra)["n_contra_face"].tolist() == [2.0]
    assert runs.neighbour_features(ns, ns, contra, exclude_experts=True)["n_contra_face"].tolist() == [1.0]


def test_vote_times_keeps_the_latest_vote_before_the_pull(tmp_path):
    """An expert who changed their vote before the pull: the later vote is the pull's (rows out of time order)."""
    times = tmp_path / "times.csv"
    pd.DataFrame({"schema": "sidewalk_a", "label_id": ["1", "1"], "label_type": "NoSidewalk", "expert_idx": 1,
                  "validation_result": ["Agree", "Disagree"], "start_timestamp": "", "source": "Validate",
                  "mission_id": 1, "label_author_is_expert": "f",
                  "end_timestamp": ["2026-01-15 00:00:00+00", "2026-01-01 00:00:00+00"]}).to_csv(times, index=False)
    t = runs.vote_times(str(times), {"sidewalk_a": "a"}, "2026-02-01")
    assert t[("a", "1", 1)][1] == "Agree"


def test_near50_counts_the_label_itself_even_when_masked():
    """The rated label always counts in its own pool, also when a sensitivity mask leaves it out."""
    d = pd.DataFrame({"face_id": ["f", "f"], "run_id": "r", "pano_id": ["p", "q"], "logit": [1.0, 3.0],
                      "latitude": [47.0, 47.0], "longitude": [-122.0, -122.0]})
    assert runs.pooled_scores(d, "near50", members=pd.Series([False, True])).tolist() == [2.0, 3.0]


def test_run_bootstrap_resamples_within_city():
    """One multi-label run per city: resampling runs within city always draws the same two runs, so every resample
    equals the observed agreement (pooled resampling would not)."""
    d = pd.DataFrame({"run_id": ["a1"] * 3 + ["b1"] * 2, "city": ["a"] * 3 + ["b"] * 2,
                      "verdict": ["correct"] * 3 + ["correct", "incorrect"]})
    assert np.allclose(runs.run_boot(d, 50, 0), 3 / 4)


def test_macro_mean_needs_ten_multi_label_runs_per_city():
    """A city with one run of 2+ is left out of the macro mean; a city with ten is in it."""
    rows = []
    for c, n_runs in (("a", runs.MIN_RUNS), ("b", 1)):
        for r in range(n_runs):
            for j in range(2):
                rows.append({"city": c, "run_id": f"{c}:f{r}:L0", "face_id": f"{c}:f{r}", "labeler_idx": r % 3,
                             "verdict": "correct" if (r + j) % 3 else "incorrect", "label_id": f"{r}{j}"})
    assert runs.macro_block(pd.DataFrame(rows), n_perm=5, n_boot=5, seed=0)["cities"] == ["a"]
