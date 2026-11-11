"""analysis/nosidewalk_gate.py on a synthetic pilot split and synthetic fold predictions, plus its job files (issue #42)."""
import doctest
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import nosidewalk_gate as gate
from calibration import _isotonic_map as im

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["label_uid", "city", "label_id", "pano_id", "face_id", "label_type", "label_class", "crop_path", "split"]


def write_pilot_split(root, n_faces=60, per_face=3, heldout_faces=12):
    """A whole-arm NoSidewalk/split.csv: 3-label faces across train/val/test, and tucson/chicago held out."""
    base = os.path.join(root, "whole", "NoSidewalk")
    os.makedirs(base, exist_ok=True)
    rows, i = [], 0
    for f in range(n_faces + heldout_faces):
        heldout = f >= n_faces
        city = ("chicago-il" if f % 2 else "tucson-az") if heldout else ("seattle-wa" if f % 3 else "amsterdam")
        split = "test_heldout" if heldout else ("train", "val", "test")[f % 3 if f % 5 else 0]
        cls = "incorrect" if f % 4 == 0 else "correct"  # both classes in every city
        for j in range(per_face):
            rows.append({"label_uid": f"{city}:{i}", "city": city, "label_id": i, "pano_id": f"p{f}_{j}",
                         "face_id": f"{city}:{f}:left", "label_type": "NoSidewalk", "label_class": cls,
                         "crop_path": f"{cls}/{city}_{i}.webp", "split": split})
            i += 1
    df = pd.DataFrame(rows, columns=COLS)
    path = os.path.join(base, "split.csv")
    df.to_csv(path, index=False)
    with open(os.path.join(root, "whole", "deployment_priors.json"), "w") as f:
        json.dump({"NoSidewalk": {"correct": 188680, "incorrect": 38485, "source": "face-size method (test)"}}, f)
    return path


def write_fold_predictions(gate_dir, seed=0, separation=2.5):
    """Predictions for every fold: the fold's test rows and test_heldout, ranking incorrect higher."""
    with open(os.path.join(gate_dir, "folds.json")) as f:
        n = json.load(f)["folds"]
    rng = np.random.default_rng(seed)
    for k in range(n):
        split = pd.read_csv(os.path.join(gate_dir, f"fold{k}", "split.csv"), dtype=str, keep_default_na=False)
        rows = split[split["split"].isin(["test", "test_heldout"])]
        y = (rows["label_class"] == "incorrect").to_numpy()
        p_inc = 1 / (1 + np.exp(-(separation * (y * 2 - 1) + rng.normal(0, 1, len(y)))))
        pd.DataFrame({"label_uid": rows["label_uid"], "fold": k, "split": rows["split"], "true": rows["label_class"],
                      "p_correct": 1 - p_inc, "p_incorrect": p_inc}).to_csv(
            gate.predictions_path(gate_dir, k), index=False)


def test_doctests():
    assert doctest.testmod(gate).failed == 0


def test_folds_group_by_face_and_leave_heldout_untouched(tmp_path):
    split = write_pilot_split(tmp_path)
    out = tmp_path / "gate"
    assert gate.main(["folds", "--split", split, "--out", str(out), "--folds", "5", "--seed", "42"]) == 0
    base = pd.read_csv(split, dtype=str, keep_default_na=False)
    folds = pd.read_csv(out / "folds.csv", dtype={"label_uid": str})
    pooled = base[base["split"].isin(["train", "val", "test"])]
    assert set(folds["label_uid"]) == set(pooled["label_uid"]) and sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    fold_of = folds.set_index("label_uid")["fold"]
    assert (pooled.assign(f=pooled["label_uid"].map(fold_of)).groupby("face_id")["f"].nunique() == 1).all()
    seen_test = []
    for k in range(5):
        v = pd.read_csv(out / f"fold{k}" / "split.csv", dtype=str, keep_default_na=False)
        assert len(v) == len(base) and set(v["split"]) == {"train", "val", "test", "test_heldout"}
        h = v[v["split"] == "test_heldout"].drop(columns="crop_path").sort_values("label_uid").reset_index(drop=True)
        hb = base[base["split"] == "test_heldout"].drop(columns="crop_path").sort_values("label_uid")
        assert h.equals(hb.reset_index(drop=True))
        assert v["crop_path"].map(os.path.isabs).all() and v["crop_path"].str.contains("/whole/NoSidewalk/").all()
        test_uids = v.loc[v["split"] == "test", "label_uid"]
        assert set(test_uids) == set(fold_of.index[fold_of == k])
        assert set(v.loc[v["split"] == "val", "label_uid"]) == set(fold_of.index[fold_of == (k + 1) % 5])
        seen_test += test_uids.tolist()
    assert sorted(seen_test) == sorted(pooled["label_uid"])  # every pooled row is test exactly once
    meta = json.loads((out / "folds.json").read_text())
    assert meta["folds"] == 5 and meta["pooled_rows"] == len(pooled) and meta["heldout_rows"] == 36
    assert meta["pooled_faces"] == pooled["face_id"].nunique()


def test_folds_refuse_fewer_than_three(tmp_path):
    split = write_pilot_split(tmp_path)
    with pytest.raises(SystemExit, match="at least 3"):
        gate.main(["folds", "--split", split, "--out", str(tmp_path / "g"), "--folds", "2"])


def test_served_out_of_fold_never_uses_the_rows_own_fold():
    """A fold whose Agree verdicts are all wrong must still be served from the other folds' (good) map."""
    rng = np.random.default_rng(1)
    n = 600
    fold = np.arange(n) % 3
    true = np.where(rng.random(n) < 0.7, "correct", "incorrect")
    p = np.clip(np.where(true == "correct", 0.95, 0.1) + rng.normal(0, 0.03, n), 0.01, 0.99)
    # fold 2: every confident "correct" verdict is actually wrong
    true = np.where((fold == 2) & (p > 0.5), "incorrect", true)
    df = pd.DataFrame({"p_correct": p, "p_incorrect": 1 - p, "true": true, "fold": fold})
    served = gate.served_out_of_fold(df, {"correct": 0.83, "incorrect": 0.17})
    agree2 = served[(served["fold"] == 2) & (served["verdict"] == "correct")]
    # the map for fold 2 was fitted on folds 0 and 1, where confident Agree verdicts are nearly all right
    assert agree2["served"].min() > 0.9
    agree0 = served[(served["fold"] == 0) & (served["verdict"] == "correct")]
    # the map for fold 0 was fitted on folds 1 and 2; fold 2 drags the Agree curve down
    assert agree0["served"].mean() < agree2["served"].mean()


def test_gate_stats_and_bar():
    df = pd.DataFrame({"verdict": ["correct"] * 4 + ["incorrect"], "served": [0.95, 0.93, 0.5, 0.99, 0.99],
                       "true": ["correct", "incorrect", "correct", "correct", "incorrect"]})
    g = gate.gate_stats(df)
    assert g["n_pass"] == 3 and g["n_agree_verdicts"] == 4 and g["precision"] == pytest.approx(2 / 3)
    assert g["coverage"] == pytest.approx(0.6)
    assert g["precision_wilson95_lower"] == pytest.approx(float(im.wilson_lower(2 / 3, 3, z=1.96)))


def test_report_end_to_end_and_splice(tmp_path):
    split = write_pilot_split(tmp_path)
    out = tmp_path / "gate"
    gate.main(["folds", "--split", split, "--out", str(out)])
    write_fold_predictions(str(out))
    doc = tmp_path / "doc.md"
    doc.write_text("head\n<!-- gate-results:start -->\nold\n<!-- gate-results:end -->\ntail\n")
    assert gate.main(["report", "--gate", str(out), "--priors", str(tmp_path / "whole" / "deployment_priors.json"),
                      "--out-json", str(out / "report" / "gate.json"), "--out-md", str(out / "report" / "gate.md"),
                      "--n-boot", "50", "--doc", str(doc)]) == 0
    rep = json.loads((out / "report" / "gate.json").read_text())
    pooled = rep["pooled"]
    assert pooled["n"] == 180 and pooled["faces"] == 60 and rep["folds"] == 5
    assert 0.5 < pooled["auc"] <= 1 and pooled["auc_ci95"][0] <= pooled["auc"] <= pooled["auc_ci95"][1]
    assert set(pooled["per_fold"]) == {"0", "1", "2", "3", "4"} and sum(g["n"] for g in pooled["per_fold"].values()) == 180
    assert set(pooled["per_city_gate"]) == {"seattle-wa", "amsterdam"}
    assert set(pooled["bar"]) == {"within_city_auc_min", "gate_precision_min", "gate_n_min", "auc_ok", "gate_ok",
                                  "passes"}
    g = pooled["agree_gate"]
    assert pooled["bar"]["gate_ok"] == (g["n_pass"] >= 100 and g["precision_wilson95_lower"] >= 0.92)
    assert pooled["within_city_n_cities"] == 2 and pooled["bar"]["auc_ok"] == (pooled["within_city_auc"] >= 0.85)
    assert pooled["bar"]["passes"] == (pooled["bar"]["auc_ok"] and pooled["bar"]["gate_ok"])
    held = rep["heldout"]
    assert set(held["per_fold"]) == {"0", "1", "2", "3", "4"} and held["mean_of_models"]["models"] == 5
    assert held["mean_of_models"]["n"] == 36 and held["per_fold"]["0"]["n"] == 36
    md = (out / "report" / "gate.md").read_text()
    assert "| pooled |" in md and "| fold 0 |" in md and "| mean of 5 models |" in md and "**Bar:**" in md
    text = doc.read_text()
    assert text.startswith("head\n<!-- gate-results:start -->\nPooled out-of-fold rows: 180") and "\nold\n" not in text
    assert text.endswith("<!-- gate-results:end -->\ntail\n")


def test_a_weak_model_does_not_pass_the_bar(tmp_path):
    split = write_pilot_split(tmp_path)
    out = tmp_path / "gate"
    gate.main(["folds", "--split", split, "--out", str(out)])
    write_fold_predictions(str(out), separation=0.3)
    gate.main(["report", "--gate", str(out), "--priors", str(tmp_path / "whole" / "deployment_priors.json"),
               "--out-json", str(out / "r.json"), "--out-md", str(out / "r.md"), "--n-boot", "20"])
    rep = json.loads((out / "r.json").read_text())
    assert rep["pooled"]["bar"]["passes"] is False and "DOES NOT PASS" in (out / "r.md").read_text()


def test_report_refuses_a_gate_without_predictions(tmp_path):
    split = write_pilot_split(tmp_path)
    out = tmp_path / "gate"
    gate.main(["folds", "--split", split, "--out", str(out)])
    with pytest.raises(SystemExit, match="no fold"):
        gate.main(["report", "--gate", str(out), "--priors", str(tmp_path / "whole" / "deployment_priors.json"),
                   "--out-json", str(out / "r.json"), "--out-md", str(out / "r.md")])


def test_predict_needs_a_model():
    with pytest.raises(SystemExit):
        gate.main(["predict", "--gate", "/g", "--fold", "0"])


@pytest.mark.parametrize("name", ["nosidewalk_gate.slurm", "nosidewalk_gate_submit.sh"])
def test_job_files_parse(name):
    subprocess.run(["bash", "-n", os.path.join(ROOT, name)], check=True)


def test_launcher_dry_run_prints_the_whole_chain(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "DATA": "/d", "GATE": str(tmp_path / "g"), "RUNS": "/r"}
    res = subprocess.run(["bash", "nosidewalk_gate_submit.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True)
    out = res.stdout.splitlines()
    assert out[0] == (f"python -u analysis/nosidewalk_gate.py folds --split /d/whole/NoSidewalk/split.csv --out "
                      f"{tmp_path / 'g'} --folds 5 --seed 42")
    sb = [line for line in out if line.startswith("sbatch")]
    assert len(sb) == 5 + 1
    assert sb[0].endswith("--job-name=ns-gate-fold0 --export=ALL,STEP=train,FOLD=0 nosidewalk_gate.slurm")
    assert sb[5].endswith("--dependency=afterany:dry1:dry2:dry3:dry4:dry5 --export=ALL,STEP=report "
                          "nosidewalk_gate.slurm")
    assert "--gpus=0" in sb[5] and all("--gpus=0" not in line for line in sb[:5])


def test_launcher_reuses_existing_folds(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    (g / "folds.json").write_text('{"folds": 3}')
    env = {**os.environ, "DRY_RUN": "1", "DATA": "/d", "GATE": str(g), "RUNS": "/r", "FOLDS": "3"}
    res = subprocess.run(["bash", "nosidewalk_gate_submit.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True)
    assert "nosidewalk_gate.py folds" not in res.stdout and f"Using the folds in {g}" in res.stderr
    assert len([line for line in res.stdout.splitlines() if line.startswith("sbatch")]) == 4


def test_job_file_steps():
    text = open(os.path.join(ROOT, "nosidewalk_gate.slurm")).read()
    assert "HF_HUB_OFFLINE" in text and "--requeue" in text and 'rm -rf "$RUN"/checkpoint-*' in text
    bad = subprocess.run(["bash", "nosidewalk_gate.slurm"], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "STEP": "x", "DATA": "/d", "GATE": "/g", "RUNS": "/r"})
    assert bad.returncode != 0
