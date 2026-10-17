"""data_generation.render_label_view through sidewalk-panorama-tools: the tilt correction centres the label.

The synthetic pano is built the way the package's own tests build one (rig frame): the label is stored at a level-frame
point, and its marker is drawn at the rig pixel ``pano_pose.corrected_pixel`` gives for camera_pitch 3, beta 0.95.
The rig pixel is chosen at a pixel centre so the square marker's centroid is exactly there.
"""
import numpy as np
import pano_pose

import data_generation as dg

W, H = 1024, 512
PITCH, BETA = 3.0, 0.95
C = dg.VIEW_SIZE / 2


def _stored_point():
    """A level-frame click whose content sits at rig pixel (300.5, 330.5) - bearing -74 deg, 26 deg below."""
    mx, my = 300.5, 330.5
    px, py = pano_pose.gravity_pixel_from_rig_pixel(mx, my, W, H, BETA * PITCH, 0.0)
    assert np.allclose(pano_pose.corrected_pixel(px, py, W, H, PITCH, 0.0, BETA), (mx, my), atol=1e-6)
    return float(px), float(py), mx, my


def _pano(mx, my, r=2):
    img = np.full((H, W, 3), 128, np.uint8)
    img[int(my) - r:int(my) + r + 1, int(mx) - r:int(mx) + r + 1] = (255, 0, 0)
    return img


def _centroid(view):
    w = np.clip(view[..., 0].astype(float) - view[..., 1].astype(float), 0, None)  # red excess, 0 on grey
    assert w.sum() > 0, "marker not in view"
    ys, xs = np.indices(w.shape)
    return (w * (xs + 0.5)).sum() / w.sum(), (w * (ys + 0.5)).sum() / w.sum()


def test_tilt_correction_centres_the_marker():
    px, py, mx, my = _stored_point()
    view, applied = dg.render_label_view(_pano(mx, my), px / W, py / H, camera_pitch=PITCH, camera_roll=None,
                                         apply_tilt=True, beta=BETA)
    assert applied and view.shape == (dg.VIEW_SIZE, dg.VIEW_SIZE, 3)
    cx, cy = _centroid(view)
    assert abs(cx - C) <= 1 and abs(cy - C) <= 1, (cx, cy)


def test_without_tilt_the_marker_is_off_centre():
    px, py, mx, my = _stored_point()
    for kwargs in ({"camera_pitch": PITCH, "apply_tilt": False},   # AI-authored rule
                   {"camera_pitch": None, "apply_tilt": True}):    # no pose known
        view, applied = dg.render_label_view(_pano(mx, my), px / W, py / H, beta=BETA, **kwargs)
        assert not applied
        cx, cy = _centroid(view)
        # 0.95 * 3 * cos(74 deg) = 0.79 deg of elevation, about 14 px at this 1024 px focal length
        assert np.hypot(cx - C, cy - C) > 8, (cx, cy)
