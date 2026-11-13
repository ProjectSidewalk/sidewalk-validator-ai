"""analysis/nosidewalk_round.py on synthetic rawLabels, a synthetic gate run and a made-up sheet (issue #42, step b)."""
import doctest
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import ai_disagree_relabel as rl
from analysis import nosidewalk_gate as gate
from analysis import nosidewalk_round as nr
from tests.test_nosidewalk_gate import write_fold_predictions, write_pilot_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ME = "00000000-0000-4000-8000-000000000001"
DESIGN = {"rater": "jon", "gate": 0.92, "pass_models": 3,
          "cities": {"chicago-il": {"phase1": 120, "phase2": 30}, "amsterdam": {"phase1": 80, "phase2": 20}},
          "shares": {"pass": 0.6, "agree_below": 0.2, "model_incorrect": 0.2}}
CONF = '''city-params {
  db-schema {
    chicago-il = "sidewalk_chicago"
    amsterdam = "sidewalk_amsterdam"
  }
  landing-page-url {
    prod {
      chicago-il = "https://c.example.edu"
      amsterdam = "https://a.example.edu"
    }
  }
}'''


def vote(user, value):
    return {"user_id": user, "validation": value, "validator_type": "Human"}


def raw_city(n, start=1000):
    """``n`` NoSidewalk labels, two per face, plus one Crosswalk row; a few rows trip each exclusion rule."""
    rows = []
    for i in range(n):
        rows.append({"label_id": str(start + i), "user_id": "u1", "pano_id": f"pano{start + i}", "pano_source": "gsv",
                     "label_type": "NoSidewalk", "severity": 3 if i % 2 else "", "tags": '["ends abruptly"]' if i % 3 == 0 else "[]",
                     "street_edge_id": i // 2, "street_side": "left", "validations": json.dumps(
                         [vote("v1", "Agree")] if i % 4 == 0 else []), "pano_x": 100, "pano_y": 50,
                     "pano_width": 1000, "pano_height": 500, "camera_pitch": 0.0, "camera_roll": 0.0,
                     "time_created": "2026-01-01", "latitude": 41.0 + i / 1e4, "longitude": -87.0})
    rows[0]["user_id"] = ME                                         # rater placed
    rows[1]["validations"] = json.dumps([vote(ME, "Unsure")])       # rater voted (Unsure counts)
    rows[2]["street_side"] = ""                                     # no face
    rows[3]["user_id"] = "51b0b927-3c8a-45b2-93de-bd878d1e5cf4"    # AI placed
    rows[4]["pano_y"] = 9999                                        # bad geometry
    rows.append(dict(rows[10], label_id=str(start + n), label_type="Crosswalk"))
    return pd.DataFrame(rows)


@pytest.fixture
def run(tmp_path):
    """A pool, a gate run and phase-1 scores on disk; returns the paths."""
    d = tmp_path / "data"
    split = write_pilot_split(d)
    g = tmp_path / "gate"
    assert gate.main(["folds", "--split", split, "--out", str(g), "--folds", "5", "--seed", "42"]) == 0
    write_fold_predictions(str(g))
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_city(300).to_csv(raw / "chicago-il.csv", index=False)
    ams = raw_city(200, start=5000)
    ams["street_edge_id"] += 1000                                           # clear of the pilot split's faces
    ams.loc[20, "street_edge_id"] = 3                                       # amsterdam:3:left is a trained face
    ams.loc[21, "pano_id"] = "p1_0"                                         # a trained panorama
    ams.to_csv(raw / "amsterdam.csv", index=False)
    ex = pd.DataFrame({"schema": ["sidewalk_chicago"] * 2 + ["sidewalk_amsterdam"], "label_id": ["1030", "1031", "5030"],
                       "author_excluded": ["t", "f", "f"]})
    ex.to_csv(tmp_path / "ex.csv", index=False)
    (tmp_path / "design.json").write_text(json.dumps(DESIGN))
    (tmp_path / "city.conf").write_text(CONF)
    return {"tmp": tmp_path, "split": split, "gate": str(g), "raw": str(raw), "ex": str(tmp_path / "ex.csv"),
            "design": str(tmp_path / "design.json"), "conf": str(tmp_path / "city.conf"),
            "priors": str(d / "whole" / "deployment_priors.json")}


def make_pool(run, monkeypatch, seed="s3cret"):
    monkeypatch.setenv("RATER_ID", ME)
    monkeypatch.setattr(rl, "inside_git_checkout", lambda p: False)
    out = run["tmp"] / "pool"
    assert nr.main(["pool", "--design", run["design"], "--rawlabels", run["raw"], "--author-excluded", run["ex"],
                    "--pilot-split", run["split"], "--cityparams", run["conf"], "--seed", seed, "--out", str(out)]) == 0
    return out


def score_pool(run, pool, seed=1):
    """A manifest and five prediction files for the phase-1 labels: p_correct high on even label ids."""
    labels = pd.read_csv(pool / "labels.csv", dtype=str)
    scored = run["tmp"] / "scored"
    scored.mkdir(exist_ok=True)
    labels.assign(status=["no_pano" if i == 0 else "ok" for i in range(len(labels))])[
        ["label_uid", "city", "status"]].to_csv(scored / "manifest.csv", index=False)
    ok = labels.iloc[1:]
    rng = np.random.default_rng(seed)
    base = np.where(ok["label_id"].astype(int) % 2 == 0, 0.97, np.where(ok["label_id"].astype(int) % 3 == 0, 0.6, 0.2))
    for k in range(5):
        p = np.clip(base + rng.normal(0, 0.01, len(ok)), 0.01, 0.99)
        pd.DataFrame({"label_uid": ok["label_uid"], "split": "test", "true": "", "p_correct": p,
                      "p_incorrect": 1 - p}).to_csv(scored / f"predictions_fold{k}.csv", index=False)
    return scored


def build(run, pool, scored, key_dir="key", seed="s3cret"):
    out = run["tmp"] / key_dir
    assert nr.main(["build", "--design", run["design"], "--pool", str(pool), "--manifest", str(scored / "manifest.csv"),
                    "--predictions", str(scored), "--gate", run["gate"], "--priors", run["priors"], "--seed", seed,
                    "--key-dir", str(out)]) == 0
    return out


def test_doctests():
    assert doctest.testmod(nr).failed == 0


def test_pool_applies_each_rule_once_and_writes_no_rater_id(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    man = json.loads((pool / "pool.json").read_text())
    chi, ams = man["cities"]["chicago-il"], man["cities"]["amsterdam"]
    assert chi["labels"] == 300 and chi["ai_placed"] == 1 and chi["no_face"] == 1 and chi["bad_geometry"] == 1
    assert chi["author_excluded"] == 1 and chi["rater_placed"] == 1 and chi["rater_voted"] == 1
    assert chi["eligible"] == 300 - 6 and chi["phase1"] == 120
    assert ams["trained_face_or_pano"] == 2 and ams["phase1"] == 80
    labels = pd.read_csv(pool / "labels.csv", dtype=str)
    assert len(labels) == 200 and labels["label_uid"].is_unique
    assert set(labels["schema"]) == {"sidewalk_chicago", "sidewalk_amsterdam"}
    eligible = pd.read_csv(pool / "eligible.csv", dtype=str)
    assert not eligible["label_uid"].isin(["chicago-il:1000", "chicago-il:1001", "chicago-il:1030"]).any()
    for name in os.listdir(pool):
        assert ME not in (pool / name).read_text()
    stage = (pool / "stage_files.txt").read_text().split()
    assert len(stage) == labels["pano_id"].nunique() and stage[0].count("/") == 2


def test_pool_is_deterministic_in_the_seed_and_a_prefix_sample(run, monkeypatch):
    a = pd.read_csv(make_pool(run, monkeypatch) / "eligible.csv", dtype=str)
    b = pd.read_csv(make_pool(run, monkeypatch) / "eligible.csv", dtype=str)
    assert a.equals(b)
    chi = a[a["city"] == "chicago-il"]
    assert chi["rank"].astype(float).is_monotonic_increasing and (chi["phase1"].astype(int).to_numpy()[:120] == 1).all()
    assert chi["phase1"].astype(int).sum() == 120


def test_pool_needs_a_rater_id(run, monkeypatch):
    monkeypatch.delenv("RATER_ID", raising=False)
    monkeypatch.setattr(rl, "inside_git_checkout", lambda p: False)
    with pytest.raises((SystemExit, ValueError)):
        nr.main(["pool", "--design", run["design"], "--rawlabels", run["raw"], "--author-excluded", run["ex"],
                 "--pilot-split", run["split"], "--cityparams", run["conf"], "--seed", "s", "--out", str(run["tmp"] / "p")])


def test_build_one_per_face_weights_and_pins(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    key_dir = build(run, pool, score_pool(run, pool))
    key = pd.read_csv(key_dir / "key.csv", dtype=str)
    man = json.loads((key_dir / "manifest.json").read_text())
    assert len(key) == 50 and key["face_id"].is_unique and key["packet_id"].is_unique
    assert list(key.columns) == nr.KEY_COLUMNS and set(key["rater"]) == {"jon"}
    assert key["label_uid"].str.startswith("sidewalk_").all() and key["label_uid_city"].str.contains("-il:|amsterdam:").all()
    assert key.groupby(["city_id", "stratum"]).size().to_dict() == {
        ("amsterdam", "agree_below"): 4, ("amsterdam", "model_incorrect"): 4, ("amsterdam", "pass"): 12,
        ("chicago-il", "agree_below"): 6, ("chicago-il", "model_incorrect"): 6, ("chicago-il", "pass"): 18}
    w = key.assign(weight=key["weight"].astype(float)).groupby("city_id")["weight"].sum()
    for city in DESIGN["cities"]:  # the weights of a city sum to its eligible population
        assert w[city] == pytest.approx(man["eligible"][city])
    assert man["scored"] == {"amsterdam": 80, "chicago-il": 119} and man["shortfall"] == {}
    assert man["key_sha256"] == rl.sha256_file(str(key_dir / "key.csv"))
    assert man["seed_sha256"] == hashlib.sha256(b"s3cret").hexdigest()
    again = build(run, pool, run["tmp"] / "scored", key_dir="key2")
    assert (again / "key.csv").read_bytes() == (key_dir / "key.csv").read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        build(run, pool, run["tmp"] / "scored")
    with pytest.raises(SystemExit, match="not the seed"):
        build(run, pool, run["tmp"] / "scored", key_dir="key3", seed="other")


def test_gallery_lists_carry_this_rounds_rule(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    key_dir = build(run, pool, score_pool(run, pool))
    out = run["tmp"] / "gallery"
    assert nr.main(["gallery", "--key", str(key_dir / "key.csv"), "--out", str(out), "--cityparams", run["conf"]]) == 0
    page = (out / "index.html").read_text()
    assert "no sidewalk on the labeled side" in page and "faded crosswalk" not in page
    urls = pd.read_csv(out / "gallery_urls.csv")
    assert urls["n"].sum() == 50 and urls["url"].str.contains("/gallery\\?labelIds=").all()
    assert "gallery" in json.loads((key_dir / "manifest.json").read_text())
    from analysis import gt_rounds as gr
    assert "faded crosswalk" in gr.RATING_RULE_HTML  # restored


def test_build_refuses_predictions_that_miss_a_crop(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    scored = score_pool(run, pool)
    p = pd.read_csv(scored / "predictions_fold2.csv", dtype=str)
    p.iloc[:-1].to_csv(scored / "predictions_fold2.csv", index=False)
    with pytest.raises(SystemExit, match="exactly"):
        build(run, pool, scored)


def test_build_refuses_a_git_checkout(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    scored = score_pool(run, pool)
    monkeypatch.setattr(rl, "inside_git_checkout", lambda p: True)
    with pytest.raises(SystemExit, match="git checkout"):
        build(run, pool, scored)


def test_short_cell_is_topped_up_from_the_same_stratum():
    design = nr.load_design(DESIGN)
    rows = []
    for city, n_pass in (("chicago-il", 5), ("amsterdam", 60)):
        for i in range(100):
            rows.append({"city": city, "label_uid": f"{city}:{i}", "face_id": f"{city}:{i}", "rank": i,
                         "stratum": "pass" if i < n_pass else ("agree_below" if i % 2 else "model_incorrect")})
    picked, short = nr.sample_phase2(pd.DataFrame(rows), design)
    sizes = picked.groupby(["city", "stratum"]).size()
    assert short == {"chicago-il/pass": 13} and sizes[("chicago-il", "pass")] == 5 and sizes[("amsterdam", "pass")] == 25
    assert len(picked) == 50 and (picked["cell_picked"] <= picked["cell_scored"]).all()


def sheet_for(key, wrong_pass=1, unsure=2):
    """Correct on pass and agree_below, incorrect on model_incorrect, with a few exceptions."""
    v = np.where(key["stratum"] == "model_incorrect", "incorrect", "correct").astype(object)
    idx = np.flatnonzero(key["stratum"] == "pass")
    v[idx[:wrong_pass]] = "incorrect"
    v[idx[wrong_pass:wrong_pass + unsure]] = "cant_tell"
    return pd.DataFrame({"packet_id": key["packet_id"], "label_type": "NoSidewalk", "verdict": v, "notes": ""})


def test_report_end_to_end(run, monkeypatch, capsys):
    pool = make_pool(run, monkeypatch)
    key_dir = build(run, pool, score_pool(run, pool))
    key = pd.read_csv(key_dir / "key.csv", dtype=str)
    sheet = run["tmp"] / "votes_jon.csv"
    sheet_for(key).to_csv(sheet, index=False)
    args = ["report", "--key", str(key_dir / "key.csv"), "--sheet", str(sheet), "--scores", str(key_dir / "scores.csv"),
            "--pilot-split", run["split"], "--n-boot", "50", "--out-json", str(run["tmp"] / "r.json"),
            "--out-md", str(run["tmp"] / "r.md")]
    with pytest.raises(SystemExit, match="not the pinned votes_jon"):
        nr.main(args)
    rl.pin(str(key_dir / "manifest.json"), "votes_jon", str(sheet), len(key))
    doc = run["tmp"] / "doc.md"
    doc.write_text(f"x\n{nr.START}\nSTALE\n{nr.END}\ny\n")
    assert nr.main(args + ["--doc", str(doc)]) == 0
    rep = json.loads((run["tmp"] / "r.json").read_text())
    p = rep["primary"]
    assert (p["k"], p["n"]) == (27, 28) and rep["cant_tell"] == 2 and not p["bar_met"]  # 28 decided is under 100
    assert (p["unsure_as_incorrect"]["k"], p["unsure_as_incorrect"]["n"]) == (27, 30)
    assert rep["strata"]["model_incorrect"]["rate"] == 0 and rep["strata"]["agree_below"]["rate"] == 1
    assert 0 < rep["population_correct"]["share"] < 1 and rep["prior_correct"] == pytest.approx(0.8306, abs=1e-3)
    assert set(rep["models"]) == {"0", "1", "2", "3", "4"} and rep["models"]["0"]["auc_weighted"] > 0.9
    assert set(rep["slices"]) == {"severity", "tag"} and "crowd_vs_expert" in rep
    text = doc.read_text()
    assert "DOES NOT PASS" in text and "STALE" not in text and text.startswith("x\n") and text.endswith("y\n")
    first = (run["tmp"] / "r.json").read_bytes()
    assert nr.main(args) == 0 and (run["tmp"] / "r.json").read_bytes() == first


def test_bar_needs_the_count_and_the_bound():
    def m(n_pass, n_wrong):
        v = ["incorrect"] * n_wrong + ["correct"] * (n_pass - n_wrong)
        return pd.DataFrame({"verdict": v + ["incorrect"] * 20, "stratum": ["pass"] * n_pass + ["model_incorrect"] * 20,
                             "city_id": "chicago-il", "weight": 1.0, "h_agree": 0, "h_dis": 0, "severity": "",
                             "tags": "[]", "pass_0": [1] * n_pass + [0] * 20,
                             "p_correct_0": [0.9] * n_pass + [0.1] * 20})
    assert nr.build_report(m(150, 3), DESIGN, 0.83, n_boot=20)["primary"]["bar_met"]
    assert not nr.build_report(m(150, 8), DESIGN, 0.83, n_boot=20)["primary"]["bar_met"]   # bound under the gate
    assert not nr.build_report(m(90, 0), DESIGN, 0.83, n_boot=20)["primary"]["bar_met"]    # too few decided


def make_neighbours(run, key_dir):
    out = run["tmp"] / "neighbours"
    assert nr.main(["neighbours", "--design", run["design"], "--rawlabels", run["raw"], "--key-dir", str(key_dir),
                    "--author-excluded", run["ex"], "--pilot-split", run["split"], "--cityparams", run["conf"],
                    "--out", str(out)]) == 0
    return out


def score_neighbours(run, nb):
    """A manifest (first label without a panorama) and five prediction files for the neighbour labels."""
    labels = pd.read_csv(nb / "labels.csv", dtype=str)
    d = run["tmp"] / "nb_scored"
    d.mkdir(exist_ok=True)
    labels.assign(status=["no_pano" if i == 0 else "ok" for i in range(len(labels))])[
        ["label_uid", "city", "status"]].to_csv(d / "manifest.csv", index=False)
    ok = labels.iloc[1:]
    p = np.where(ok["label_id"].astype(int) % 2 == 0, 0.97, 0.2)
    for k in range(5):
        pd.DataFrame({"label_uid": ok["label_uid"], "split": "test", "true": "", "p_correct": p,
                      "p_incorrect": 1 - p}).to_csv(d / f"predictions_fold{k}.csv", index=False)
    return d


def test_neighbours_are_the_other_labels_on_the_rated_faces(run, monkeypatch):
    pool = make_pool(run, monkeypatch)
    scored = score_pool(run, pool)
    key_dir = build(run, pool, scored)
    key = pd.read_csv(key_dir / "key.csv", dtype=str)
    scores = pd.read_csv(key_dir / "scores.csv", dtype=str)
    nb = make_neighbours(run, key_dir)
    faces = pd.read_csv(nb / "faces.csv", dtype=str)
    labels = pd.read_csv(nb / "labels.csv", dtype=str)
    raw = pd.concat([pd.read_csv(os.path.join(run["raw"], f"{c}.csv"), dtype=str).assign(city=c)
                     for c in DESIGN["cities"]])
    raw = raw[raw["label_type"] == "NoSidewalk"]
    raw_face = raw["city"] + ":" + raw["street_edge_id"] + ":" + raw["street_side"]
    assert set(faces["face_id"]) == set(key["face_id"]) and faces["label_uid"].is_unique
    assert len(faces) == int(raw_face.isin(set(key["face_id"])).sum())          # every label on a rated face
    assert set(faces.loc[faces["rated"] == "1", "label_uid"]) == set(key["label_uid_city"])
    new = faces[faces["new"] == "1"]
    assert set(labels["label_uid"]) == set(new["label_uid"]) and len(new) > 0
    assert not new["label_uid"].isin(scores["label_uid"]).any()                 # phase-1 labels are not scored again
    assert (faces.loc[faces["scored"] == "1", "label_uid"].isin(scores["label_uid"])).all()
    # a rule that keeps a label out of the pool is a flag here, except the two that make a crop impossible or moot
    assert (new["ai_placed"] == "0").all() and (new["bad_geometry"] == "0").all()
    assert set(nr.NEIGHBOUR_FLAGS) <= set(faces.columns) and "user_id" not in faces and "user_id" not in labels
    assert (faces["same_labeler"] == "1").mean() > 0.9                           # the fixture has one labeler
    man = json.loads((nb / "neighbours.json").read_text())
    assert man["labels"] == len(labels) and man["labels_sha256"] == rl.sha256_file(str(nb / "labels.csv"))
    assert sum(c["new"] for c in man["cities"].values()) == len(labels)
    assert len((nb / "stage_files.txt").read_text().split()) == labels["pano_id"].nunique()
    for name in os.listdir(nb):
        assert ME not in (nb / name).read_text()
    monkeypatch.setattr(rl, "inside_git_checkout", lambda p: True)
    with pytest.raises(SystemExit, match="inside a git checkout"):
        make_neighbours(run, key_dir)


def test_pool_is_unchanged_by_the_crop_labels_refactor(run, monkeypatch):
    """``crop_labels`` and ``write_stage_files`` were factored out of ``pool``; the files must be what the inline
    code wrote (the real pinned pool was also reproduced byte for byte when this changed)."""
    pool = make_pool(run, monkeypatch)
    labels = pd.read_csv(pool / "labels.csv", dtype=str, keep_default_na=False)
    assert list(labels.columns) == nr.ale.LABEL_COLUMNS + ["schema", "face_id", "rank"]
    assert (labels["label_class"] == "").all() and (labels["n_votes"].astype(int)
                                                    == labels["human_agree"].astype(int)
                                                    + labels["human_disagree"].astype(int)).all()
    stage = (pool / "stage_files.txt").read_text().split()
    assert stage == sorted(set(stage)) and all(p.endswith(".jpg") for p in stage)


def test_neighbour_scores_reproduces_the_pinned_scores_first(run, monkeypatch, capsys):
    pool = make_pool(run, monkeypatch)
    scored = score_pool(run, pool)
    key_dir = build(run, pool, scored)
    nb = make_neighbours(run, key_dir)
    d = score_neighbours(run, nb)
    out_json = run["tmp"] / "nb.json"
    args = ["neighbour-scores", "--design", run["design"], "--key-dir", str(key_dir), "--gate", run["gate"],
            "--priors", run["priors"], "--pinned-predictions", str(scored), "--neighbours", str(nb),
            "--manifest", str(d / "manifest.csv"), "--predictions", str(d), "--out-json", str(out_json),
            "--out-md", str(run["tmp"] / "nb.md")]
    assert nr.main(args) == 0
    fs = pd.read_csv(nb / "face_scores.csv", dtype=str, keep_default_na=False)
    faces = pd.read_csv(nb / "faces.csv", dtype=str)
    assert len(fs) == len(faces) and "user_id" not in fs
    new = fs[fs["new"] == "1"]
    assert (new["crop_status"] == "no_pano").sum() == 1 and (new.loc[new["crop_status"] == "no_pano", "stratum"] == "").all()
    ok = new[new["crop_status"] == "ok"]
    assert (ok["stratum"] == np.where(ok["label_id"].astype(int) % 2 == 0, "pass", "model_incorrect")).all()
    pinned = pd.read_csv(key_dir / "scores.csv", dtype=str).set_index("label_uid")
    old = fs[fs["crop_status"] == "phase1"].set_index("label_uid")
    assert len(old) > 0 and (old["stratum"] == pinned.loc[old.index, "stratum"]).all()
    rep = json.loads(out_json.read_text())
    assert rep["all"]["labels"] == int((fs["stratum"] != "").sum()) and set(rep["slices"]) == {"severity", "tag"}
    assert set(rep["cities"]) == set(DESIGN["cities"]) and "label_uid" not in out_json.read_text()
    f = rep["faces_with_2plus_scored"]
    assert f["faces"] == f["all_pass"] + f["none_pass"] + f["mixed"]
    assert "Mean share per face" in (run["tmp"] / "nb.md").read_text()
    # a prediction file that does not cover the ok crops, then a different gate run, both stop the run
    short = pd.read_csv(d / "predictions_fold2.csv")
    short.iloc[1:].to_csv(d / "predictions_fold2.csv", index=False)
    with pytest.raises(SystemExit, match="does not cover exactly"):
        nr.main(args)
    short.to_csv(d / "predictions_fold2.csv", index=False)
    gp = os.path.join(run["gate"], "fold0", "predictions.csv")
    g = pd.read_csv(gp)
    g.assign(p_correct=g["p_correct"] * 0.999).to_csv(gp, index=False)
    with pytest.raises(SystemExit, match="not the gate run the key was built with"):
        nr.main(args)


def test_design_file_and_job_file():
    d = nr.load_design(os.path.join(ROOT, "analysis", "gt_rounds", "nosidewalk_b.json"))
    q = nr.quotas(d)
    assert sum(q.values()) == 250 and sum(n for (c, s), n in q.items() if s == "pass") == 150
    subprocess.run(["bash", "-n", os.path.join(ROOT, "nosidewalk_round.slurm")], check=True)
    text = open(os.path.join(ROOT, "nosidewalk_round.slurm")).read()
    assert "--fixed-crop-half-px 1024" in text and "HF_HUB_OFFLINE" in text and "--requeue" in text
