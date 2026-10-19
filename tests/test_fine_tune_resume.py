"""Resume guard in fine_tune_classifier.py: what counts as a checkpoint, and when a resume is refused."""
import json
import os
from types import SimpleNamespace

import pytest

from fine_tune_classifier import resume_checkpoint, run_key

ARGS = SimpleNamespace(seed=42, epochs=25, best_metric="macro_f1")


def _key(tmp_path, **overrides):
    split_dir = tmp_path / "data" / "CurbRamp"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split.csv").write_text("label_uid,split\nseattle-wa:1,train\n")
    return run_key("CurbRamp", str(tmp_path / "data"), SimpleNamespace(**{**vars(ARGS), **overrides}))


def _checkpoint(out, step, complete=True):
    d = out / f"checkpoint-{step}"
    d.mkdir(parents=True)
    if complete:
        (d / "trainer_state.json").write_text("{}")


def test_fresh_run_writes_the_key(tmp_path):
    out = tmp_path / "dinov2"
    key = _key(tmp_path)
    assert resume_checkpoint(str(out), key) is None
    assert json.loads((out / "run_key.json").read_text()) == key


def test_resumes_from_the_latest_complete_checkpoint(tmp_path):
    out = tmp_path / "dinov2"
    key = _key(tmp_path)
    resume_checkpoint(str(out), key)
    _checkpoint(out, 164)
    _checkpoint(out, 1312)   # numeric, not lexical, order: 1312 > 328
    _checkpoint(out, 328)
    _checkpoint(out, 1476, complete=False)   # cut off mid-save: no trainer_state.json
    assert resume_checkpoint(str(out), key) == os.path.join(str(out), "checkpoint-1312")


@pytest.mark.parametrize("change", [{"epochs": 8}, {"seed": 1}, {"best_metric": "loss"}])
def test_refuses_checkpoints_from_other_settings(tmp_path, change):
    out = tmp_path / "dinov2"
    resume_checkpoint(str(out), _key(tmp_path))
    _checkpoint(out, 164)
    with pytest.raises(SystemExit, match="another run"):
        resume_checkpoint(str(out), _key(tmp_path, **change))


def test_refuses_checkpoints_from_another_split(tmp_path):
    out = tmp_path / "dinov2"
    resume_checkpoint(str(out), _key(tmp_path))
    _checkpoint(out, 164)
    (tmp_path / "data" / "CurbRamp" / "split.csv").write_text("label_uid,split\nseattle-wa:2,train\n")
    key = run_key("CurbRamp", str(tmp_path / "data"), ARGS)
    with pytest.raises(SystemExit, match="differs"):
        resume_checkpoint(str(out), key)


def test_refuses_checkpoints_without_a_key(tmp_path):
    out = tmp_path / "dinov2"
    _checkpoint(out, 164)   # e.g. a run from before run_key.json existed
    with pytest.raises(SystemExit, match="no run_key.json"):
        resume_checkpoint(str(out), _key(tmp_path))


def test_incomplete_checkpoints_and_a_stale_key_start_fresh(tmp_path):
    out = tmp_path / "dinov2"
    out.mkdir()
    (out / "run_key.json").write_text('{"stale": true}')
    _checkpoint(out, 164, complete=False)
    key = _key(tmp_path)
    assert resume_checkpoint(str(out), key) is None
    assert json.loads((out / "run_key.json").read_text()) == key


def test_a_torn_trainer_state_is_not_complete(tmp_path):
    out = tmp_path / "dinov2"
    key = _key(tmp_path)
    resume_checkpoint(str(out), key)
    _checkpoint(out, 164)
    _checkpoint(out, 328, complete=False)
    (out / "checkpoint-328" / "trainer_state.json").write_text('{"best_glo')   # cut off mid-write
    assert resume_checkpoint(str(out), key) == os.path.join(str(out), "checkpoint-164")


def test_mismatch_message_names_only_the_changed_fields(tmp_path):
    out = tmp_path / "dinov2"
    resume_checkpoint(str(out), _key(tmp_path))
    _checkpoint(out, 164)
    with pytest.raises(SystemExit) as e:
        resume_checkpoint(str(out), _key(tmp_path, epochs=8))
    assert "'epochs': (25, 8)" in str(e.value) and "split_sha256" not in str(e.value)
