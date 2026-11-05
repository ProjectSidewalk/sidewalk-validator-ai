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


def test_no_workers_unless_the_start_method_is_fork(monkeypatch):
    import multiprocessing
    import fine_tune_classifier
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: "spawn")
    assert fine_tune_classifier.default_num_workers() == 0
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: "fork")
    assert 0 <= fine_tune_classifier.default_num_workers() <= 8


# --- --context-arm / --batch-size (issue #14, context_model.py) ------------------------------------------------------

def _ctx(**kw):
    return SimpleNamespace(**{**vars(ARGS), "context_arm": "none", "context_view_res": 224, "batch_size": 8, **kw})


def test_default_context_options_leave_the_key_unchanged(tmp_path):
    import fine_tune_classifier as ft
    assert ft.context_settings(_ctx()) == {} and ft.context_settings(ARGS) == {}
    assert _key(tmp_path, context_arm="none", context_view_res=224, batch_size=8) == _key(tmp_path)
    assert ft.extra_frame_columns(_ctx()) == () and ft.extra_frame_columns(ARGS) == ()


def test_context_arm_view_res_and_batch_size_are_in_the_key(tmp_path):
    import fine_tune_classifier as ft
    keys = [_key(tmp_path, **vars(_ctx(**kw))) for kw in
            ({}, {"context_arm": "two_scale"}, {"context_arm": "view_heatmap"},
             {"context_arm": "view_heatmap", "context_view_res": 448}, {"context_arm": "side_info"},
             {"batch_size": 4})]
    assert len({json.dumps(k, sort_keys=True) for k in keys}) == len(keys)
    assert ft.context_settings(_ctx(context_arm="two_scale", batch_size=4)) == {
        "context_arm": "two_scale", "context_view_res": 224, "batch_size": 4}


def test_extra_frame_columns_per_arm():
    import context_model
    import fine_tune_classifier as ft
    for arm in context_model.ARMS:
        cols = ft.extra_frame_columns(_ctx(context_arm=arm))
        assert cols == context_model.FRAME_COLUMNS[arm] and {"label_uid", "split"} <= set(cols)
    assert "crop_half_px" in ft.extra_frame_columns(_ctx(context_arm="view_heatmap"))
    assert "tags" in ft.extra_frame_columns(_ctx(context_arm="side_info"))


@pytest.mark.parametrize("argv,match", [
    (["--context-arm", "two_scale"], "--output-dir"),
    (["--batch-size", "5"], "divide"),
    (["--context-arm", "view_heatmap", "--output-dir", "x", "--context-view-res", "300"], "invalid choice"),
])
def test_context_options_are_checked_before_any_import(argv, match, capsys):
    import fine_tune_classifier as ft
    with pytest.raises(SystemExit):
        ft.main(["--label_type", "CurbRamp", *argv])
    assert match in capsys.readouterr().err


def test_load_split_frames_carries_extra_columns(tmp_path):
    import fine_tune_classifier as ft
    d = tmp_path / "Toy"
    d.mkdir()
    (d / "split.csv").write_text("label_uid,label_class,crop_path,split,crop_half_px\n"
                                 "c:1,correct,a.webp,train,200\nc:2,incorrect,b.webp,val,300\n")
    train, val = ft.load_split_frames("Toy", str(tmp_path), extra_columns=("label_uid", "split", "crop_half_px"))
    assert list(train.columns) == ["image", "label", "label_uid", "split", "crop_half_px"]
    assert val.crop_half_px.tolist() == [300] and val.split.tolist() == ["val"]
    train, _ = ft.load_split_frames("Toy", str(tmp_path))
    assert list(train.columns) == ["image", "label"]
