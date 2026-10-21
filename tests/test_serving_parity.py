"""Train/serve parity (issue #5): the training crop pipeline against sidewalk-ai-api's, on synthetic panoramas.

Two groups:

* Always on (CPU, numpy only, what CI runs): the pure helpers of ``analysis/serving_parity.py``, and the training
  stages (``read_store_panorama``, ``render_label_view``, ``create_perspective_crop``, ``save_webp``) with a stub depth
  model, checked byte for byte against the serving recipe copied verbatim from sidewalk-ai-api's ``main.py``
  (:func:`api_crop_recipe`). The copy can go stale; the second group checks the live code.
* Only with ``SIDEWALK_AI_API_DIR`` pointing at a sidewalk-ai-api checkout (and torch + flask installed): stages 1, 2
  and 4 against that checkout's own code, ``/process`` included, through Flask's test client::

      SIDEWALK_AI_API_DIR=~/Git/sidewalk-ai-api-wt/tilt-fields pytest -q tests/test_serving_parity.py

The real-panorama, real-model measurement is ``analysis/serving_parity.py`` (docs/serving-parity-2026-10.md).
"""
import doctest
import os
import sys

import numpy as np
import pytest

import data_generation as dg
from analysis import serving_parity as sp

API_DIR = os.environ.get("SIDEWALK_AI_API_DIR")


def structured_pano(width):
    """A (width/2, width, 3) uint8 equirect whose content changes at every pixel and differs per channel (two ramps
    and a checkerboard; sidewalk-ai-api's tests/test_geometry.py builds the same one), so any channel swap, shift or
    rounding difference shows."""
    h = width // 2
    yy, xx = np.mgrid[0:h, 0:width]
    return np.stack([(xx * 7) % 256, (yy * 5) % 256, ((xx // 8 + yy // 8) % 2) * 255], -1).astype(np.uint8)


class StubDepth:
    """Records the image ``infer_image`` receives; returns a constant-offset gradient (``depth_at_centre`` metres at
    the centre of a 2048 view), so an indexing slip changes the crop."""

    def __init__(self, depth_at_centre=20.0):
        self.d0, self.inputs = depth_at_centre, []

    def infer_image(self, image):
        self.inputs.append(np.array(image))
        h, w = image.shape[:2]
        rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
        return (self.d0 + 0.001 * (cols - w // 2) + 0.002 * (rows - h // 2)).astype(np.float32)

    # sidewalk-ai-api's interface for the same model
    def predict_depth(self, image):
        return self.infer_image(image)


def api_crop_recipe(view_bgr, depth):
    """sidewalk-ai-api ``main.py`` ``/process``, from the depth map to the PIL image the validator gets, copied verbatim
    (head 174a169; only the names of the inputs changed). Returns the RGB array of that PIL image."""
    import cv2
    from PIL import Image
    perspective_image = view_bgr
    center_x, center_y = perspective_image.shape[1] // 2, perspective_image.shape[0] // 2
    img_h, img_w = perspective_image.shape[:2]

    inv_depth = 1 / depth[center_y, center_x]
    crop_size_half = int(inv_depth * 6100)

    start_x = max(0, center_x - crop_size_half)
    start_y = max(0, center_y - crop_size_half)
    end_x = min(img_w, center_x + crop_size_half)
    end_y = min(img_h, center_y + crop_size_half)

    crop_height, crop_width = perspective_image[start_y:end_y, start_x:end_x].shape[:2]
    if crop_width > crop_height:
        resize_width, resize_height = 640, int(640 * crop_height / crop_width)
    else:
        resize_width, resize_height = int(640 * crop_width / crop_height), 640
    perspective_image = cv2.resize(perspective_image[start_y:end_y, start_x:end_x], (resize_width, resize_height),
                                   interpolation=cv2.INTER_CUBIC)

    perspective_image = cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)
    perspective_image = Image.fromarray(perspective_image)
    return np.asarray(perspective_image)


POSES = [(None, None), (3.0, None), (-5.0, 1.0)]  # the poses both sides read the same way
POINTS = [(0.3, 0.6), (0.001, 0.62), (0.75, 0.9)]


# --- pure helpers ----------------------------------------------------------------------------------------------------

def test_doctests():
    res = doctest.testmod(sp)
    assert res.failed == 0 and res.attempted > 0


def test_compare_arrays():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    assert sp.compare_arrays(a, b)["identical"]
    b[0, 0, 2] = 3
    r = sp.compare_arrays(a, b, centre=4)
    assert not r["identical"] and r["max_abs"] == 3 and r["centre_max_abs"] == 0
    assert r["frac_px_diff"] == pytest.approx(1 / 100) and r["frac_diff"] == pytest.approx(1 / 300)
    b[5, 5, 0] = 1
    assert sp.compare_arrays(a, b, centre=4)["centre_max_abs"] == 1
    assert not sp.compare_arrays(a, a.astype(np.int16))["identical"]  # same values, different dtype
    assert np.isnan(sp.compare_arrays(a, a[:5])["max_abs"])


def test_crop_half_rules_disagree_only_by_rounding():
    """Training rounds 6100 / depth in float64, the API in float32: the same int nearly everywhere, but not always."""
    assert sp.crop_half_train(np.float32(10.0)) == sp.crop_half_api(np.float32(10.0)) == 610
    depths, bad = sp.crop_half_sweep(1.0013, 1.0014)
    assert bad.any(), "the sweep should find the float32 rounding cases near 6100 / 1.0013"
    d = depths[bad][0]
    assert sp.crop_half_api(d) - sp.crop_half_train(d) == 1
    # and the API's rule in this helper is the API's code: same answer as the verbatim recipe's crop size
    view = np.zeros((2048, 2048, 3), np.uint8)
    depth = np.full((2048, 2048), d, np.float32)
    assert api_crop_recipe(view, depth).shape == (640, 640, 3)
    n, idx = sp.crop_half_disagreements(depths)
    assert n == int(bad.sum()) and idx == list(np.flatnonzero(bad))


def test_vote_parity_counts_vote_changes():
    amap = {"correct": {"0.5": 0.5, "0.9": 0.95}, "incorrect": {"0.5": 0.6, "0.8": 0.93}}
    ref = [[0.95, 0.05], [0.15, 0.85], [0.55, 0.45], [0.95, 0.05]]
    new = [[0.95, 0.05], [0.25, 0.75], [0.45, 0.55], [0.05, 0.95]]
    r = sp.vote_parity(ref, new, amap)
    # row 2: incorrect vote lost; row 3: verdict flips, neither votes; row 4: correct vote becomes an incorrect vote
    assert (r["verdict_flips"], r["vote_changes"], r["votes_ref"], r["votes"]) == (2, 2, 3, 2)
    assert sp.served_vote([0.5, 0.5], amap)[0] == "correct"  # the API's argmax keeps the first index on a tie


def test_api_lookup_matches_calibration():
    from calibration._isotonic_map import api_lookup
    amap = {"0.5": 0.1, "0.75": 0.5, "0.99": 0.97}
    for c in (0.4, 0.5, 0.8, 0.995, 1.0):
        assert sp.api_lookup(amap, c) == api_lookup(amap, c)


def test_load_points_roundtrip(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("point_id,x_norm,y_norm,note\nA,0.25,0.6,x\n")
    assert sp.load_points(str(f)) == [("A", 0.25, 0.6, "x")]
    assert len(sp.load_points()) == 10


# --- training stages vs the serving recipe (CPU, numpy render, stub depth) -------------------------------------------

def _store(tmp_path, img):
    import cv2
    store = sp.make_store(str(tmp_path / "store"), str(tmp_path / "src.jpg"))
    assert cv2.imwrite(str(tmp_path / "src.jpg"), img)
    return store


def test_read_store_panorama_is_bgr_and_the_api_read():
    """Stage 1: the training read is cv2.imread + resize(8192x4096, INTER_LINEAR), BGR, byte for byte the API's
    cache-path recipe (``Panorama.__init__``), and a blue left half stays in channel 0."""
    import tempfile
    import cv2
    img = np.zeros((256, 512, 3), np.uint8)
    img[:, :256] = (255, 0, 0)      # BGR blue
    img[:, 256:] = (0, 0, 255)      # BGR red
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        store = _store(Path(d), img)
        got, info = dg.read_store_panorama(store, sp.STORE_CITY, sp.STORE_PANO_ID)
        path = dg.pano_store_path(store, sp.STORE_CITY, sp.STORE_PANO_ID)
        api = cv2.resize(cv2.imread(path), (8192, 4096), interpolation=cv2.INTER_LINEAR)
    assert got.shape == (4096, 8192, 3) and (info["pano_native_w"], info["pano_native_h"]) == (512, 256)
    assert np.array_equal(got, api)
    assert got[2000, 1000, 0] > 200 and got[2000, 1000, 2] < 50       # blue is channel 0: BGR
    assert got[2000, 7000, 2] > 200 and got[2000, 7000, 0] < 50


@pytest.fixture(scope="module")
def pano_rgb():
    return structured_pano(2048)


@pytest.mark.parametrize("pitch, roll", POSES)
@pytest.mark.parametrize("x, y", POINTS)
def test_training_crop_is_the_api_crop(pano_rgb, x, y, pitch, roll):
    """Stages 3-4: create_perspective_crop hands depth the BGR view, and its crop equals, byte for byte, what the API's
    code cuts from that BGR view and converts to RGB for the validator."""
    view_rgb, _ = dg.render_label_view(pano_rgb, x, y, pitch, roll, beta=0.95)
    depth = StubDepth(20.0)
    crop, info = dg.create_perspective_crop(depth, pano_rgb, x, y, camera_pitch=pitch, camera_roll=roll, beta=0.95)
    assert np.array_equal(depth.inputs[0], view_rgb[..., ::-1]), "depth must see the BGR view, as the API passes it"
    served = api_crop_recipe(view_rgb[..., ::-1], depth.infer_image(view_rgb[..., ::-1]))
    assert sp.compare_arrays(crop, served)["identical"]
    assert info["crop_half_px"] == int(6100 / 20.0)
    # a channel-order slip on either side would not pass silently
    assert not sp.compare_arrays(crop, served[..., ::-1])["identical"]


@pytest.mark.parametrize("d0", [2.0, 3.5, 61.37, 79.9])
def test_near_and_far_crops_match_including_clipping(pano_rgb, d0):
    """A near feature (crop_half > 1024: the square is clipped at the frame) and far ones."""
    view_rgb, _ = dg.render_label_view(pano_rgb, 0.4, 0.7)
    depth = StubDepth(d0)
    crop, info = dg.create_perspective_crop(depth, pano_rgb, 0.4, 0.7)
    assert info["clipped"] == int(6100 / d0 > 1024)
    assert np.array_equal(crop, api_crop_recipe(view_rgb[..., ::-1], depth.infer_image(view_rgb[..., ::-1])))


def test_round_depth_is_the_crop_half_skew(pano_rgb):
    """The one crop-geometry difference in the code: at a depth where 6100 / depth is an integer (61.0 m -> 100) the
    API's float32 ``1 / d * 6100`` lands just under it and truncates to 99, training's float64 gives 100. The crop
    then differs everywhere (a 199 px square resized to 640 instead of 201 px). About 6e-5 of float32 depths in
    1-80 m hit this (``serving_parity.py crop-half``)."""
    view_rgb, _ = dg.render_label_view(pano_rgb, 0.4, 0.7)
    depth = StubDepth(61.0)
    crop, info = dg.create_perspective_crop(depth, pano_rgb, 0.4, 0.7)
    d = depth.infer_image(view_rgb[..., ::-1])
    assert info["crop_half_px"] == sp.crop_half_train(d[1024, 1024]) == 100
    assert sp.crop_half_api(d[1024, 1024]) == 99
    assert not np.array_equal(crop, api_crop_recipe(view_rgb[..., ::-1], d))


def test_webp_is_the_one_lossy_step(tmp_path, pano_rgb):
    """Stage 5: training models see a decoded q95 WebP, the API the raw crop. PNG round-trips exactly; WebP does not."""
    crop, _ = dg.create_perspective_crop(StubDepth(20.0), pano_rgb, 0.3, 0.6)
    sp.save_png(crop, str(tmp_path / "c.png"))
    assert np.array_equal(sp.load_png(str(tmp_path / "c.png")), crop)
    decoded = sp.webp_roundtrip(crop, str(tmp_path / "c.webp"))
    r = sp.compare_arrays(crop, decoded)
    assert not r["identical"] and 0 < r["mean_abs"] < 10


# --- against a sidewalk-ai-api checkout (SIDEWALK_AI_API_DIR) -------------------------------------------------------

needs_api = pytest.mark.skipif(not API_DIR, reason="set SIDEWALK_AI_API_DIR to a sidewalk-ai-api checkout")


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    """sidewalk-ai-api's ``main`` and ``panorama`` modules, imported from ``SIDEWALK_AI_API_DIR`` as its own tests do
    (cwd = the checkout, for its relative ``accuracy_mappings/`` and ``API_VERSION`` reads)."""
    pytest.importorskip("torch")
    pytest.importorskip("flask")
    os.environ.setdefault("SIDEWALK_AI_API_KEY", "serving-parity-test")
    old = os.getcwd()
    os.chdir(API_DIR)
    sys.path.insert(0, API_DIR)
    try:
        import main
        from sidewalk_ai_api import panorama
    finally:
        os.chdir(old)
    return main, panorama


@pytest.fixture(scope="module")
def pano_8192():
    return structured_pano(8192)


@needs_api
def test_api_stage1_cache_read(api, tmp_path):
    _, panorama = api
    img = structured_pano(1024)[..., ::-1].copy()
    store = _store(tmp_path, img)
    path = dg.pano_store_path(store, sp.STORE_CITY, sp.STORE_PANO_ID)
    served = panorama.Panorama(sp.STORE_PANO_ID, cached_image_path=path)
    ours, _ = dg.read_store_panorama(store, sp.STORE_CITY, sp.STORE_PANO_ID)
    assert served.image_source == "cache"
    assert np.array_equal(served.panorama_image, ours)


@needs_api
@pytest.mark.parametrize("pitch, roll", POSES)
@pytest.mark.parametrize("x, y", POINTS[:2])
def test_api_stage2_render(api, pano_8192, x, y, pitch, roll):
    """The API's render_view_torch and training's, both on the cpu, from the same BGR pano: byte-identical; the numpy
    renderer (training's --allow-cpu path) is within one level."""
    from sidewalk_panorama_tools.geometry import label_view
    _, panorama = api
    bgr = np.ascontiguousarray(pano_8192[..., ::-1])
    view, tilt = label_view(x, y, 90, 8192 // 4, pitch, roll, 0.95)
    served = panorama.render_view_torch(bgr, view, tilt)
    if served is None or served.shape != (2048, 2048, 3):
        pytest.fail("unexpected API render")
    ours, _ = dg.render_label_view(pano_8192, x, y, pitch, roll, beta=0.95, device="cpu")
    assert np.array_equal(served, ours[..., ::-1])
    ours_np, _ = dg.render_label_view(pano_8192, x, y, pitch, roll, beta=0.95)
    assert np.abs(ours_np.astype(int) - ours.astype(int)).max() <= 1


def _process_crop(api, monkeypatch, tmp_path, pano_bgr, x, y, pitch, roll):
    """The PIL image the API's validator receives from ``/process`` for this label, as an RGB array."""
    import cv2
    main, _ = api
    store = sp.make_store(str(tmp_path / "store"), str(tmp_path / "src.png"))
    assert cv2.imwrite(str(tmp_path / "src.png"), pano_bgr)  # lossless; the store path ends .jpg, imread sniffs
    seen = []

    class Validator:
        def validate(self, image):
            seen.append(np.asarray(image))
            return "correct", 0.5

    class Tagger:
        def inference(self, image):
            return [], [], {}

    monkeypatch.setattr(main, "SCRAPES_DIR", store)
    monkeypatch.chdir(API_DIR)
    models = main.Models(depth=StubDepth(20.0), taggers={t: Tagger() for t in main.TAGGER_LABEL_TYPES},
                         validators={t: Validator() for t in main.VALIDATOR_LABEL_TYPES})
    client = main.create_app(models).test_client()
    data = {"label_type": "curbramp", "panorama_id": sp.STORE_PANO_ID, "city": sp.STORE_CITY,
            "x": repr(x), "y": repr(y), "tilt_beta": "0.95"}
    if pitch is not None:
        data["camera_pitch"] = repr(pitch)
    if roll is not None:
        data["camera_roll"] = repr(roll)
    resp = client.post("/process", data=data, headers={"Authorization": f"Bearer {main.API_KEY}"})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["image_source"] == "cache"
    return seen[0]


def _training_crop(store_dir, x, y, pitch, roll):
    pano_bgr, _ = dg.read_store_panorama(store_dir, sp.STORE_CITY, sp.STORE_PANO_ID)
    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])  # data_generation.main
    crop, _ = dg.create_perspective_crop(StubDepth(20.0), pano_rgb, x, y, camera_pitch=pitch, camera_roll=roll,
                                         beta=0.95, device="cpu")
    return crop


@needs_api
@pytest.mark.parametrize("pitch, roll", POSES)
def test_api_stage4_process_crop_is_the_training_crop(api, monkeypatch, tmp_path, pano_8192, pitch, roll):
    """End to end up to the validator's input: store -> /process -> PIL crop vs store -> create_perspective_crop
    (torch cpu render, as a real run renders on the GPU) with the same stub depth: byte-identical."""
    bgr = np.ascontiguousarray(pano_8192[..., ::-1])
    served = _process_crop(api, monkeypatch, tmp_path, bgr, 0.3, 0.6, pitch, roll)
    ours = _training_crop(str(tmp_path / "store"), 0.3, 0.6, pitch, roll)
    assert sp.compare_arrays(served, ours)["identical"]


@needs_api
@pytest.mark.xfail(strict=True, reason="known skew: the API applies no tilt when camera_roll is sent without "
                                       "camera_pitch, training tilts by the roll (docs/serving-parity-2026-10.md); "
                                       "GSV labels carry no roll, so no real label hits it. Remove the xfail when "
                                       "either side changes.")
def test_api_roll_without_pitch(api, monkeypatch, tmp_path, pano_8192):
    bgr = np.ascontiguousarray(pano_8192[..., ::-1])
    served = _process_crop(api, monkeypatch, tmp_path, bgr, 0.3, 0.6, None, 1.0)
    ours = _training_crop(str(tmp_path / "store"), 0.3, 0.6, None, 1.0)
    assert sp.compare_arrays(served, ours)["identical"]
