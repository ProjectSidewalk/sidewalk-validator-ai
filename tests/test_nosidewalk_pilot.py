"""analysis/nosidewalk_pilot.py on synthetic runs, and the pilot's job file and launcher (issue #42)."""
import doctest
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import nosidewalk_pilot as pilot
from calibration import _isotonic_map as im

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_run(data, arm, seed, with_map=True, separation=2.0):
    """A split of 60 test and 30 tucson-az rows on 3-label faces, and predictions that rank incorrect higher."""
    rng = np.random.default_rng(seed)
    base = os.path.join(data, arm, "NoSidewalk")
    os.makedirs(os.path.join(base, "calibration", f"s{seed}"), exist_ok=True)
    rows = []
    for i in range(90):
        split = "test" if i < 60 else "test_heldout"
        city = ("seattle-wa" if i % 2 else "chicago-il") if split == "test" else "tucson-az"
        rows.append({"label_uid": f"{city}:{i}", "city": city, "pano_id": f"p{i}", "face_id": f"{city}:{i // 3}:left",
                     "label_class": "incorrect" if (i // 3) % 3 == 0 else "correct", "split": split})
    split = pd.DataFrame(rows)
    split.to_csv(os.path.join(base, "split.csv"), index=False)
    y = (split.label_class == "incorrect").to_numpy()
    logit = separation * (y * 2 - 1) + rng.normal(0, 1, len(y))
    p_inc = 1 / (1 + np.exp(-logit))
    pd.DataFrame({"label_uid": split.label_uid, "split": split.split, "true": split.label_class,
                  "p_correct": 1 - p_inc, "p_incorrect": p_inc}).to_csv(
        os.path.join(base, f"predictions_s{seed}.csv"), index=False)
    if with_map:
        api_map = {c: {im.threshold_key(t): round(float(t), 4) for t in im.GRID} for c in im.API_CLASSES}
        with open(os.path.join(base, "calibration", f"s{seed}", "nosidewalk.json"), "w") as f:
            json.dump(api_map, f)


def test_agree_gate_uses_the_map_when_given():
    d = pd.DataFrame({"p_correct": [0.99, 0.95], "p_incorrect": [0.01, 0.05], "true": ["correct", "correct"]})
    low = {"correct": {"0.5": 0.5, "0.99": 0.93}, "incorrect": {"0.5": 0.5}}
    g = pilot.agree_gate(d, low)
    assert (g["rule"], g["n_pass"], g["coverage"], g["precision"]) == ("map", 1, 0.5, 1.0)
    assert 0 < g["precision_wilson95_lower"] < 1
    none = pilot.agree_gate(d.assign(p_correct=[0.6, 0.6], p_incorrect=[0.4, 0.4]))
    assert none["n_pass"] == 0 and np.isnan(none["precision"])


def test_join_predictions_groups_by_face_then_panorama():
    split = pd.DataFrame({"label_uid": ["a:1", "a:2"], "city": ["a", "a"], "pano_id": ["p", "q"],
                          "face_id": ["a:9:left", ""]})
    pred = pd.DataFrame({"label_uid": ["a:1", "a:2"], "split": ["test"] * 2, "true": ["correct"] * 2,
                         "p_correct": [0.9, 0.9], "p_incorrect": [0.1, 0.1]})
    assert pilot.join_predictions(split, pred)["group"].tolist() == ["a:9:left", "pano:q"]
    with pytest.raises(ValueError):
        pilot.join_predictions(split.iloc[:1], pred)


def test_collect_summarise_markdown_and_splice(tmp_path):
    data = str(tmp_path / "data")
    for arm in ("depth", "whole"):
        for seed in (42, 43):
            write_run(data, arm, seed, with_map=(arm == "depth"))
    rows, missing = pilot.collect(data, seeds=(42, 43, 44), n_boot=50)
    assert missing == ["depth/s44", "whole/s44"] and len(rows) == 8
    r = rows[0]
    assert (r["arm"], r["seed"], r["split"], r["n"], r["n_incorrect"], r["faces"]) == ("depth", 42, "test", 60, 21, 40)
    assert r["auc"] > 0.8 and r["auc_ci95"][0] <= r["auc"] <= r["auc_ci95"][1]
    assert r["within_city_n_cities"] == 2 and r["agree_gate"]["rule"] == "map"
    assert rows[4]["agree_gate"]["rule"] == "raw"
    held = [x for x in rows if x["split"] == "test_heldout"][0]
    assert np.isnan(held["within_city_auc"]) or held["within_city_n_cities"] == 1
    summary = pilot.summarise(rows)
    assert [(s["arm"], s["split"], s["runs"]) for s in summary] == [
        ("depth", "test", 2), ("depth", "test_heldout", 2), ("whole", "test", 2), ("whole", "test_heldout", 2)]
    md = pilot.markdown(summary, rows, missing)
    assert "| depth | test | 2 | 60 (21) |" in md and "depth/s44" in md
    doc = tmp_path / "doc.md"
    doc.write_text(f"# x\n{pilot.START}\n_placeholder_\n{pilot.END}\nafter\n")
    out_json = tmp_path / "rep" / "pilot.json"
    pilot.main(["--data", data, "--seeds", "42,43", "--n-boot", "20", "--out-json", str(out_json),
                "--out-md", str(tmp_path / "rep" / "pilot.md"), "--doc", str(doc)])
    text = doc.read_text()
    assert "_placeholder_" not in text and text.endswith(f"{pilot.END}\nafter\n") and "| whole | test |" in text
    assert json.loads(out_json.read_text())["missing"] == []
    with pytest.raises(ValueError):
        pilot.splice("no markers", "x")
    with pytest.raises(SystemExit):
        pilot.main(["--data", str(tmp_path / "empty"), "--out-json", str(out_json), "--out-md", str(tmp_path / "m")])


def test_doctests():
    assert doctest.testmod(pilot).failed == 0


@pytest.mark.parametrize("name", ["nosidewalk_pilot.slurm", "nosidewalk_submit.sh"])
def test_job_files_parse(name):
    subprocess.run(["bash", "-n", os.path.join(ROOT, name)], check=True)


def _dry_run(tmp_path, **env):
    full = {**os.environ, "DRY_RUN": "1", "DATA": "/d", "RUNS": "/r", "PANO_STORE": "/p", **env}
    full.pop("EXPERTS", None)
    return subprocess.run(["bash", "nosidewalk_submit.sh"], cwd=ROOT, env=full, capture_output=True, text=True,
                          check=True)


def test_launcher_dry_run_prints_the_whole_chain(tmp_path):
    res = _dry_run(tmp_path)
    out = res.stdout.splitlines()
    assert "python -u analysis/fetch_all.py /d/rawlabels-ns.tmp --label-types NoSidewalk" in out
    sb = [line for line in out if line.startswith("sbatch")]
    assert len(sb) == 2 + 6 + 1
    assert "--export=ALL,STEP=data,ARM=depth nosidewalk_pilot.slurm" in sb[0]
    assert "--dependency=afterok:dry1 --export=ALL,STEP=train,ARM=depth,SEED=42" in sb[1]
    assert "--export=ALL,STEP=train,ARM=whole,SEED=44" in sb[7] and "afterok:dry5" in sb[7]
    assert sb[8].endswith("--dependency=afterany:dry2:dry3:dry4:dry6:dry7:dry8 --export=ALL,STEP=report "
                          "nosidewalk_pilot.slurm")
    assert "EXPERTS is not set" in res.stderr


def test_launcher_reuses_a_complete_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".complete").touch()
    res = _dry_run(tmp_path, CACHE=str(cache), SEEDS="7", ARMS="whole")
    assert "fetch_all" not in res.stdout and f"Using the rawLabels cache {cache}" in res.stderr
    assert len([line for line in res.stdout.splitlines() if line.startswith("sbatch")]) == 3


def test_job_file_steps_and_arms():
    text = open(os.path.join(ROOT, "nosidewalk_pilot.slurm")).read()
    for needle in ("--verdict-unit face", "--group-col face_id", "--fixed-crop-half-px 1024",
                   "calibration.nosidewalk_prior", "--hold-back incorrect", "--tag \"s$SEED\"", "#SBATCH --account=makelab",
                   "#SBATCH --partition=gpu-l40s"):
        assert needle in text
    bad = subprocess.run(["bash", "nosidewalk_pilot.slurm"], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "STEP": "nonsense", "DATA": "/d"})
    assert bad.returncode == 2 and "STEP must be" in bad.stderr
