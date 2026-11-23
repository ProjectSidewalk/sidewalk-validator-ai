"""fine_tune_classifier.py's run provenance (issue 65): timestamps, Slurm ids and the git commit in
``training_run.json``, and the model card's sentence built from them."""
import doctest
import os
import subprocess

import pytest

import fine_tune_classifier as ft
import hf_cards


def _git(cwd, *args):
    subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})


def test_doctests():
    assert doctest.testmod(ft).failed == 0


def test_utc_now_is_iso_8601_z():
    now = ft.utc_now()
    assert len(now) == 20 and now.endswith("Z") and now[10] == "T"


def test_git_state_reports_the_head_and_dirty_tracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "one")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert ft.git_state(str(tmp_path)) == {"git_commit": head, "git_dirty": False}
    (tmp_path / "f.txt").write_text("b\n")
    assert ft.git_state(str(tmp_path)) == {"git_commit": head, "git_dirty": True}
    (tmp_path / "f.txt").write_text("a\n")
    (tmp_path / "untracked.txt").write_text("x\n")  # untracked files are not "dirty": they cannot change the code
    assert ft.git_state(str(tmp_path))["git_dirty"] is False


def test_git_state_is_none_outside_a_checkout(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.delenv("GIT_DIR", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert ft.git_state(str(plain)) == {"git_commit": None, "git_dirty": None}


def test_git_state_default_is_this_checkout():
    state = ft.git_state()
    # In a checkout the commit is a 40-hex SHA; in a bare export (no .git) both are None. Either way, no exception.
    assert state["git_commit"] is None or len(state["git_commit"]) == 40


def test_git_state_keeps_the_commit_when_status_fails(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "one")
    real_run = subprocess.run

    def slow_status(cmd, **kw):
        if "status" in cmd:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return real_run(cmd, **kw)

    monkeypatch.setattr(ft.subprocess, "run", slow_status)
    state = ft.git_state(str(tmp_path))
    assert len(state["git_commit"]) == 40 and state["git_dirty"] is None


def test_slurm_state_reads_the_job_environment(monkeypatch):
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(name, raising=False)
    assert ft.slurm_state() == {"slurm_job_id": None, "slurm_array_job_id": None, "slurm_array_task_id": None,
                                "slurm_restart_count": None}
    monkeypatch.setenv("SLURM_JOB_ID", "42918974")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "42918970")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "")
    assert ft.slurm_state() == {"slurm_job_id": 42918974, "slurm_array_job_id": 42918970, "slurm_array_task_id": 3,
                                "slurm_restart_count": None}
    assert ft.slurm_state({"SLURM_JOB_ID": "not-a-number"})["slurm_job_id"] is None


def test_run_provenance_records_the_start_state_and_a_changed_finish(monkeypatch):
    started = {"git_commit": "a" * 40, "git_dirty": False, "slurm_job_id": 7, "slurm_array_job_id": None,
               "slurm_array_task_id": None, "slurm_restart_count": 2}
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": "a" * 40, "git_dirty": False})
    same = ft.run_provenance("2020-01-01T00:00:00Z", started, None)
    assert same["git_commit"] == "a" * 40 and same["slurm_restart_count"] == 2 and same["resumed_from"] is None
    assert same["started_at"] <= same["finished_at"] and "git_state_at_finish" not in same
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": "b" * 40, "git_dirty": True})
    moved = ft.run_provenance("2020-01-01T00:00:00Z", started, "/runs/checkpoint-40")
    assert moved["git_commit"] == "a" * 40  # the code that was imported, not the pulled HEAD
    assert moved["git_state_at_finish"] == {"git_commit": "b" * 40, "git_dirty": True}
    assert moved["resumed_from"] == "checkpoint-40"
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": "a" * 40, "git_dirty": None})
    assert "git_state_at_finish" not in ft.run_provenance("2020-01-01T00:00:00Z", started, None)  # status unknown


def test_publish_code_commit_uses_the_shared_helper(monkeypatch):
    import publish_to_hf as pub
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": "c" * 40, "git_dirty": True})
    assert pub.code_commit() == "c" * 40 + "-dirty"
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": "c" * 40, "git_dirty": None})
    assert pub.code_commit() == "c" * 40 + "-dirty?"  # status failed: not reported as clean
    monkeypatch.setattr(ft, "git_state", lambda *a, **k: {"git_commit": None, "git_dirty": None})
    assert pub.code_commit() is None


def test_card_training_sentence(monkeypatch):
    assert doctest.testmod(hf_cards).failed == 0
    run = {"label_type": "CurbRamp", "seed": 42, "epochs": 25, "finished_at": "2026-11-11T17:05:09Z",
           "git_commit": "54e6d21" + "0" * 33, "git_dirty": False, "slurm_job_id": 42918974}
    ctx = {"label_type": "CurbRamp", "model_repo": "m", "dataset_repo": "d", "dataset": {}, "code_commit": None,
           "overrides": {}, "training_run": run}
    card = hf_cards.model_card(ctx)
    assert "Trained 2026-11-11 (UTC) at sidewalk-validator-ai commit `54e6d21`, Slurm job 42918974." in card
    assert "start and finish time (UTC), Slurm job id, git commit" in card
    no_git = hf_cards.model_card({**ctx, "training_run": {k: v for k, v in run.items() if not k.startswith("git")}})
    assert "Trained 2026-11-11 (UTC), Slurm job 42918974." in no_git
    old = hf_cards.model_card({**ctx, "training_run": {"label_type": "CurbRamp", "seed": 42, "epochs": 25}})
    assert "Trained 20" not in old and "Slurm job 4" not in old  # the file-table row still names the fields


# ---------------------------------------------------------------------------------------------------------------------
# Issue 3: a label-level (leaky) split is marked, recorded, and never trained into the production directory
# ---------------------------------------------------------------------------------------------------------------------

def _split_file(path, mode=None):
    import pandas as pd
    df = pd.DataFrame({"label_uid": ["a:1", "a:2"], "label_class": ["correct", "incorrect"],
                       "crop_path": ["c/a_1.webp", "i/a_2.webp"]})
    if mode is not None:
        df["split_mode"] = mode
        df["split_grouped"] = ["train", "test"]
    df["split"] = ["train", "train"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def test_split_mode_of_reads_the_marker_column(tmp_path):
    assert ft.split_mode_of(_split_file(tmp_path / "g.csv")) is None
    assert ft.split_mode_of(_split_file(tmp_path / "l.csv", "label_level")) == "label_level"
    assert ft.split_mode_of(str(tmp_path / "absent.csv")) is None


def test_split_mode_is_recorded_only_for_a_marked_split(tmp_path):
    assert ft.split_settings(_split_file(tmp_path / "g.csv")) == {}
    assert ft.split_settings(_split_file(tmp_path / "l.csv", "label_level")) == {"split_mode": "label_level"}


def test_a_marked_split_is_refused_at_the_default_output_dir_before_torch(tmp_path):
    data = tmp_path / "data"
    leaky = _split_file(tmp_path / "runs" / "leaky.csv", "label_level")
    # explicitly pointed at the production directory
    with pytest.raises(SystemExit) as e:
        ft.main(["--label_type", "CurbRamp", "--data-dir", str(data), "--split", leaky,
                 "--output-dir", str(data / "CurbRamp" / "dinov2")])
    assert "label-level" in str(e.value)
    # a marked file copied over the production split.csv, with the default output dir
    _split_file(data / "CurbRamp" / "split.csv", "label_level")
    with pytest.raises(SystemExit) as e:
        ft.main(["--label_type", "CurbRamp", "--data-dir", str(data)])
    assert "label-level" in str(e.value)
