"""context_model.py (issue #14): side features, normaliser, heatmap and point mapping (always run), and the three
context arms of the DINOv2 classifier (torch + transformers; skipped on the CPU-only CI). Tiny Dinov2Config backbones
are built from a config and saved locally, never downloaded."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest

import context_model as cm


def test_doctests():
    r = doctest.testmod(cm)
    assert r.attempted > 0 and r.failed == 0


def side_df():
    return pd.DataFrame({
        "label_uid": ["c:1", "c:2", "c:3", "c:4"],
        "severity": [3, None, 1, "2"],
        "tags": ['["a","b"]', "a;b", None, "[]"],
        "image_capture_date": ["2020-09", "2024-09-15", "bad", None],
        "depth_m": [10.0, None, 0, "5.5"],
        "high_quality_user": [True, False, None, "True"],
        "clipped": [0, 1, None, "1"],
    })


def test_side_feature_frame():
    f = cm.side_feature_frame(side_df())
    assert list(f.columns) == cm.SIDE_COLUMNS
    assert f.severity.isna().tolist() == [False, True, False, False] and f.severity_missing.tolist() == [0, 1, 0, 0]
    assert f.n_tags.tolist() == [2, 2, 0, 0]
    assert f.age_years.iloc[0] == pytest.approx(4.0, abs=1e-3) and f.age_years.iloc[1] == 0
    assert f.age_missing.tolist() == [0, 0, 1, 1]
    assert f.log_depth_m.iloc[0] == pytest.approx(np.log(10)) and f.log_depth_m.iloc[3] == pytest.approx(np.log(5.5))
    assert f.depth_missing.tolist() == [0, 1, 1, 0]   # None and 0 are both "no depth"
    assert f.high_quality_user.tolist() == [1, 0, 0, 1] and f.clipped.tolist() == [0, 1, 0, 1]
    # an explicit reference date is used as given (the config's, at inference)
    assert cm.side_feature_frame(side_df(), "2025-09").age_years.iloc[1] == pytest.approx(1.0, abs=1e-2)


def test_normaliser_floors_fills_and_round_trips_through_json():
    f = cm.side_feature_frame(side_df())
    stats = cm.fit_normaliser(f)
    i_clip = cm.SIDE_COLUMNS.index("clipped")
    assert stats["std"][i_clip] > 0
    const = f.assign(high_quality_user=1.0)
    s2 = cm.fit_normaliser(const)
    assert s2["std"][cm.SIDE_COLUMNS.index("high_quality_user")] == 1e-6   # zero std floored
    x = cm.normalise(f, stats)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    i_sev = cm.SIDE_COLUMNS.index("severity")
    med = stats["median"][i_sev]
    assert x[1, i_sev] == pytest.approx((med - stats["mean"][i_sev]) / stats["std"][i_sev])  # NaN -> median
    back = json.loads(json.dumps(stats))
    assert np.array_equal(cm.normalise(f, back), x)
    with pytest.raises(ValueError, match="columns"):
        cm.normalise(f, {**stats, "columns": stats["columns"][::-1]})


def test_all_missing_column_gets_median_zero():
    f = cm.side_feature_frame(side_df().assign(severity=None))
    stats = cm.fit_normaliser(f)
    assert stats["median"][cm.SIDE_COLUMNS.index("severity")] == 0.0
    assert np.isfinite(cm.normalise(f, stats)).all()


def test_heatmap_peak_symmetry_and_sigma():
    m = cm.heatmap(31, 41, 20, 15, 3.0)
    assert m.shape == (31, 41) and m.max() == 1.0 and np.unravel_index(m.argmax(), m.shape) == (15, 20)
    assert m[15, 23] == pytest.approx(np.exp(-0.5)) and m[18, 20] == pytest.approx(np.exp(-0.5))
    assert np.allclose(m[:, 20 - 5], m[:, 20 + 5])
    a = cm.heatmap(31, 41, 20, 15, (6.0, 3.0))
    assert a[15, 26] == pytest.approx(np.exp(-0.5)) and a[18, 20] == pytest.approx(np.exp(-0.5))


def test_heatmap_sigma_scale_and_floor():
    assert cm.heatmap_sigma(1024, 224) == 56.0          # a whole-view crop: a quarter of the input
    assert cm.heatmap_sigma(0, 224) == cm.HEATMAP_SIGMA_FLOOR
    assert cm.heatmap_sigma(400, 448) == 2 * cm.heatmap_sigma(400, 224)
    assert cm.heatmap_sigma(30, 224, floor=1.0) == pytest.approx(30 * 224 / 2048 / 2)


def test_map_point_identity_and_flip():
    assert cm.map_point(5, 7, (0, 0, 20, 20), 20) == (5.0, 7.0)
    assert cm.map_point(5, 7, (0, 0, 20, 20), 20, flip=True) == (14.0, 7.0)


def test_frame_columns_cover_what_the_arms_read():
    assert set(cm.FRAME_COLUMNS) == set(cm.ARMS)
    for cols in cm.FRAME_COLUMNS.values():
        assert {"label_uid", "split"} <= set(cols)
    assert {"severity", "tags", "image_capture_date", "depth_m", "high_quality_user", "clipped"} <= set(
        cm.FRAME_COLUMNS["side_info"])


def test_point_mapping_matches_torchvision_resized_crop():
    """A small blob moved by F.resized_crop (and a flip), as the training transform calls it, lands where map_point
    says (its intensity centroid; a blob rather than one pixel, which a downsampling resize can step over)."""
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchvision.transforms.functional")
    s, res = 448, 224
    for (cy, cx), crop, flip in [((223, 223), (40, 60, 360, 300), False), ((223, 223), (0, 0, 448, 448), True),
                                 ((100, 300), (50, 180, 220, 260), True)]:
        img = torch.from_numpy(cm.heatmap(s, s, cx, cy, 4.0))[None]
        out = F.resized_crop(img, *crop, [res, res], antialias=True)
        if flip:
            out = F.hflip(out)
        w = out[0].numpy()
        yy, xx = np.mgrid[0:res, 0:res]
        got = (float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum()))
        want = cm.map_point(cx, cy, crop, res, flip)
        assert got == pytest.approx(want, abs=0.6), (crop, flip)
