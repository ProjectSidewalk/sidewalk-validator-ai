"""context_crops.py (issue #14): the wide crop and downsampled view, the resumable index, the data_generation hook
and the backfill of an existing dataset. CPU only (the numpy renderer)."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import context_crops as cc
import data_generation as dg


def gradient_pano(w=2048, h=1024):
    """An RGB pano with structure in both directions, so a shifted or mis-scaled crop is visible."""
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
    return np.concatenate([np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)),
                           np.full((h, w, 1), 90, np.float32)], axis=2).astype(np.uint8)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def test_doctests():
    r = doctest.testmod(cc)
    assert r.attempted > 0 and r.failed == 0


@pytest.mark.parametrize("half,factor,want", [(100, 4, (400, False)), (300, 4, (1024, True)),
                                              (300, 2, (600, False)), (256, 4, (1024, False)),
                                              (1500, 4, (1024, True))])
def test_wide_half_px(half, factor, want):
    assert cc.wide_half_px(half, factor) == want


def test_wide_half_px_rejects_zero():
    with pytest.raises(ValueError):
        cc.wide_half_px(0)


def test_downsample_view_shape_and_flat_stays_flat():
    flat = np.full((2048, 2048, 3), (12, 200, 77), np.uint8)
    out = cc.downsample_view(flat)
    assert out.shape == (cc.VIEW_PX, cc.VIEW_PX, 3)
    assert (out == np.array([12, 200, 77], np.uint8)).all()  # no border artefacts from the area filter
    assert cc.downsample_view(flat, 224).shape == (224, 224, 3)
    with pytest.raises(ValueError, match="square"):
        cc.downsample_view(np.zeros((10, 20, 3), np.uint8))


def test_write_context_files_paths_and_geometry(tmp_path):
    pano = gradient_pano()
    info = cc.write_context(pano, 0.4, 0.55, 300, str(tmp_path), "correct/toy_1", camera_pitch=2.0)
    assert info == {"wide_path": "wide/correct/toy_1.webp", "view_path": "view/correct/toy_1.webp",
                    "wide_half_px": 1024, "wide_clipped": 1}
    wide, view = read_rgb(tmp_path / info["wide_path"]), read_rgb(tmp_path / info["view_path"])
    assert view.shape == (cc.VIEW_PX, cc.VIEW_PX, 3)
    assert wide.shape == (dg.CROP_LONG_SIDE, dg.CROP_LONG_SIDE, 3)
    # Both are centred on the label: their centre pixels agree (up to WebP and resampling).
    c_w, c_v = wide[wide.shape[0] // 2, wide.shape[1] // 2], view[cc.VIEW_PX // 2, cc.VIEW_PX // 2]
    assert np.abs(c_w.astype(int) - c_v.astype(int)).max() <= 6
    # A capped wide crop IS the whole view: the downsampled wide crop and the view look the same.
    import cv2
    small = cv2.resize(wide, (cc.VIEW_PX, cc.VIEW_PX), interpolation=cv2.INTER_AREA)
    assert np.abs(small.astype(int) - view.astype(int)).mean() < 3


def test_write_context_matches_the_training_crop_geometry(tmp_path):
    """The wide crop at factor 1 is the training crop: same view, same centre_crop."""
    pano = gradient_pano()
    view, _ = dg.render_label_view(pano, 0.3, 0.5, 1.0, None, beta=0.95)
    tight, _, _ = dg.centre_crop(view, 200)
    info = cc.write_context(pano, 0.3, 0.5, 200, str(tmp_path), "s", camera_pitch=1.0, factor=1)
    assert info["wide_half_px"] == 200 and info["wide_clipped"] == 0
    assert np.abs(read_rgb(tmp_path / info["wide_path"]).astype(int) - tight.astype(int)).mean() < 2


def test_write_context_is_deterministic(tmp_path):
    pano = gradient_pano()
    a = cc.write_context(pano, 0.6, 0.52, 150, str(tmp_path / "a"), "x")
    b = cc.write_context(pano, 0.6, 0.52, 150, str(tmp_path / "b"), "x")
    assert a == b and a["wide_half_px"] == 600 and a["wide_clipped"] == 0
    for k in ("wide_path", "view_path"):
        assert (tmp_path / "a" / a[k]).read_bytes() == (tmp_path / "b" / b[k]).read_bytes()


def _touch(out, info):
    for k in ("wide_path", "view_path"):
        p = out / info[k]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def _info(stem):
    return {"wide_path": f"wide/{stem}.webp", "view_path": f"view/{stem}.webp", "wide_half_px": 400,
            "wide_clipped": 0}


def test_index_appends_reloads_and_refuses_duplicates(tmp_path):
    out = tmp_path / "context"
    with cc.ContextIndex(str(out)) as ctx:
        for uid, stem in (("c:1", "correct/c_1"), ("c:2", "incorrect/c_2")):
            _touch(out, _info(stem))
            ctx.write(uid, _info(stem))
        with pytest.raises(ValueError, match="already"):
            ctx.write("c:1", _info("correct/c_1"))
        # flushed per row: readable while still open
        assert list(pd.read_csv(out / cc.INDEX_FILE).label_uid) == ["c:1", "c:2"]
    again = cc.ContextIndex(str(out))
    assert again.has("c:1") and len(again) == 2
    with pytest.raises(ValueError, match="already"):
        again.write("c:2", _info("incorrect/c_2"))
    again.close()
    assert list(pd.read_csv(out / cc.INDEX_FILE).columns) == cc.INDEX_COLUMNS


def test_index_drops_rows_outside_keep_or_without_files(tmp_path):
    out = tmp_path / "context"
    with cc.ContextIndex(str(out)) as ctx:
        for i in (1, 2, 3):
            if i != 3:
                _touch(out, _info(f"correct/c_{i}"))
            ctx.write(f"c:{i}", _info(f"correct/c_{i}"))
    with open(out / cc.INDEX_FILE, "a") as f:
        f.write("c:4,wide/x.webp\n")  # a torn last line after a crash
    ctx = cc.ContextIndex(str(out), keep={"c:1", "c:3", "c:4"})
    assert ctx.uids == {"c:1"} and ctx.dropped == 3  # c:2 not kept, c:3 has no files, c:4 is torn
    ctx.close()
    assert list(pd.read_csv(out / cc.INDEX_FILE).label_uid) == ["c:1"]


def test_load_index_requires_uids_and_files(tmp_path):
    base = tmp_path / "Lt"
    out = base / "context"
    with cc.ContextIndex(str(out)) as ctx:
        for i in (1, 2):
            _touch(out, _info(f"correct/c_{i}"))
            ctx.write(f"c:{i}", _info(f"correct/c_{i}"))
    df = cc.load_index(str(base), require_uids=["c:1", "c:2"])
    assert df.wide_path.str.startswith(str(out)).all()
    with pytest.raises(ValueError, match=r"c:9"):
        cc.load_index(str(base), require_uids=["c:1", "c:9"])
    (out / "view" / "correct" / "c_2.webp").unlink()
    with pytest.raises(ValueError, match=r"missing file.*c:2"):
        cc.load_index(str(base), require_uids=["c:2"])
    with pytest.raises(FileNotFoundError, match="backfill"):
        cc.load_index(str(tmp_path / "nothing"))
    assert len(cc.index_sha256(str(base))) == 64


def test_pose_mismatch_rules():
    import crop_pose
    row = {"pano_source": "gsv", "pano_id": "p", "camera_pitch": "2.0", "camera_roll": "", "author": "human",
           "pose_source": "gsv_db", "pose_pitch": "2.0", "pose_roll": "", "click_lat_offset_deg": "0.0",
           "tilt_beta": "0.95"}
    pose = crop_pose.pose_for_row(row, 0.95)
    assert cc.pose_mismatch(row, pose, 0.95) is None
    assert "pose_pitch" in cc.pose_mismatch({**row, "pose_pitch": "1.0"}, pose, 0.95)
    assert "pose_source" in cc.pose_mismatch({**row, "pose_source": "none"}, pose, 0.95)
    assert "tilt_beta" in cc.pose_mismatch({**row, "tilt_beta": "0.5"}, pose, 0.95)
    old = {k: v for k, v in row.items() if k not in cc.POSE_CHECK}  # a manifest from before #72
    assert cc.pose_mismatch(old, pose, 0.95) is None
    assert "before #72" in cc.pose_mismatch({**old, "pano_source": "mapillary"}, pose, 0.95)
