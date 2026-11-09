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


# ---------------------------------------------------------------------------------------------------------------------
# Region conditioning and --init-from (analysis/region_transfer.py, issue #33)
# ---------------------------------------------------------------------------------------------------------------------

REGION_DEFAULTS = {"region_conditioning": "none", "region_dropout": 0.15, "region_dim": 16, "cities_json": None,
                   "init_from": None}


def test_default_run_key_is_unchanged_by_region_and_init_flags(tmp_path):
    from fine_tune_classifier import init_settings, region_settings
    old = _key(tmp_path)
    new = _key(tmp_path, **REGION_DEFAULTS)
    assert new == old
    assert set(old) == {"label_type", "split_sha256", "seed", "epochs", "best_metric", "base_checkpoint"}
    assert region_settings(SimpleNamespace(**REGION_DEFAULTS)) == {} == init_settings(SimpleNamespace(**REGION_DEFAULTS))


def _init_dir(tmp_path, name, payload):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(payload)
    return str(d)


def test_region_and_init_settings_enter_the_key(tmp_path):
    cities = tmp_path / "cities.json"
    cities.write_text('{"cities": []}')
    cond = {**REGION_DEFAULTS, "region_conditioning": "country", "cities_json": str(cities)}
    key = _key(tmp_path, **cond)
    assert key["region_conditioning"] == "country" and key["region_dropout"] == 0.15 and "cities_json_sha256" in key
    assert "cities_json" not in key   # paths stay out of the key, as vote_weights does
    out = tmp_path / "dinov2"
    resume_checkpoint(str(out), key)
    _checkpoint(out, 164)
    assert resume_checkpoint(str(out), _key(tmp_path, **cond)) == os.path.join(str(out), "checkpoint-164")
    with pytest.raises(SystemExit, match="region_dropout"):
        resume_checkpoint(str(out), _key(tmp_path, **{**cond, "region_dropout": 0.3}))

    a = _init_dir(tmp_path, "ckpt-a", b"weights a")
    b = _init_dir(tmp_path, "ckpt-b", b"weights b")
    out2 = tmp_path / "ft"
    key_a = _key(tmp_path, **{**REGION_DEFAULTS, "init_from": a})
    assert "init_from" not in key_a and len(key_a["init_from_sha256"]) == 64
    resume_checkpoint(str(out2), key_a)
    _checkpoint(out2, 10)
    with pytest.raises(SystemExit, match="init_from_sha256"):
        resume_checkpoint(str(out2), _key(tmp_path, **{**REGION_DEFAULTS, "init_from": b}))


def test_init_weights_are_hashed_once(tmp_path, monkeypatch):
    import hashlib

    import fine_tune_classifier
    fine_tune_classifier._weights_sha256.cache_clear()
    calls = []
    real = hashlib.sha256

    def counting(*a):
        calls.append(1)
        return real(*a)
    a = _init_dir(tmp_path, "ckpt-a", b"weights a")
    monkeypatch.setattr(fine_tune_classifier.hashlib, "sha256", counting)
    args = SimpleNamespace(**{**REGION_DEFAULTS, "init_from": a})
    first, second = fine_tune_classifier.init_settings(args), fine_tune_classifier.init_settings(args)
    assert first == second and first["init_from_sha256"] == real(b"weights a").hexdigest()
    assert len(calls) == 1
    fine_tune_classifier._weights_sha256.cache_clear()


def test_new_flags_are_validated(tmp_path):
    import fine_tune_classifier
    with pytest.raises(SystemExit):
        fine_tune_classifier.main(["--label_type", "X", "--region-conditioning", "country"])   # no --cities-json
    with pytest.raises(SystemExit):
        fine_tune_classifier.main(["--label_type", "X", "--init-from", str(tmp_path)])        # no config.json


def test_extra_columns_pass_through_load_split_frames(tmp_path):
    from fine_tune_classifier import extra_frame_columns, load_split_frames
    d = tmp_path / "Toy"
    d.mkdir()
    (d / "split.csv").write_text("label_uid,city,label_class,crop_path,split\n"
                                 "a:1,cdmx,correct,c/1.webp,train\na:2,spgg,incorrect,c/2.webp,val\n")
    train, val = load_split_frames("Toy", str(tmp_path), extra_columns=("city",))
    assert list(train.columns) == ["image", "label", "city"] and val["city"].tolist() == ["spgg"]
    plain, _ = load_split_frames("Toy", str(tmp_path))
    assert list(plain.columns) == ["image", "label"]
    assert extra_frame_columns(SimpleNamespace()) == ()
    assert extra_frame_columns(SimpleNamespace(region_conditioning="region")) == ("city",)
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
