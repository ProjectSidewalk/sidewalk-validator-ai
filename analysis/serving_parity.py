"""Train/serve parity (issue #5): do training and sidewalk-ai-api cut the same crop and tensor for the same label?

Runs both pipelines on one real panorama and a set of label points, stage by stage, and measures every difference:

1. pano read + resize (``data_generation.read_store_panorama`` vs the API's ``Panorama`` cache path)
2. view render (training numpy ``render_view``, training ``render_view_torch`` on cpu and mps, the API's
   ``render_view_torch`` on cpu)
3. depth at the view centre (Depth Anything V2 metric vkitti ViT-B, same checkpoint, both on BGR; plus two probes:
   the view with its channels swapped, and the same model on mps)
4. ``crop_half_px``, ``clipped`` and the crop array
5. WebP q95: training models see a decoded WebP, serving sees the raw crop
6. the preprocessed tensor: transformers 4.55 ``use_fast=False`` vs 5.x ``backend="pil"`` on the same PIL crop, with
   the fast (torchvision) processors as a control
7. the five production validators end to end: logits, probabilities, verdict and served vote

The training side runs **the training code itself** (``data_generation.render_label_view`` /
``create_perspective_crop`` / ``depth_sized_crop`` / ``save_webp``, ``evaluate.run_inference``'s preprocessing), and
the serving side runs **sidewalk-ai-api's real ``/process`` handler** through Flask's test client, with the real
``Panorama`` cache path, the real depth model and the real ``ImageValidator`` classes; only recorders are wrapped
around the depth model and the validator, so the crop the validator receives is captured, not re-derived. The two
sides need different package pins, so each runs as a worker in its own interpreter and they exchange ``.npy`` /
``.png`` / JSON files; ``compare`` then builds the tables.

Stages (subcommands)::

    # 1. training side, in a venv with the training pins (requirements.txt or environment.yml's)
    .venv/bin/python analysis/serving_parity.py train-worker --pano sample_pano.jpg --depth-checkpoint vitb.pth \\
        --api-dir ~/Git/sidewalk-ai-api-wt/tilt-fields --out WORK/train --models
    # 2. serving side, in a venv with sidewalk-ai-api's requirements.txt (reads WORK/train for the stage-6 inputs)
    api-venv/bin/python analysis/serving_parity.py api-worker --api-dir ~/Git/sidewalk-ai-api-wt/tilt-fields \\
        --depth-checkpoint vitb.pth --train-dir WORK/train --out WORK/api --models
    # 3. tables (any venv with numpy + pandas)
    .venv/bin/python analysis/serving_parity.py compare --train WORK/train --api WORK/api \\
        --api-dir ~/Git/sidewalk-ai-api-wt/tilt-fields --out docs/serving-parity-2026-10
    # ``run`` does all three, given the two interpreters:
    .venv/bin/python analysis/serving_parity.py run --train-python .venv/bin/python --api-python api-venv/bin/python \\
        --api-dir ... --pano ... --depth-checkpoint ... --work WORK --out docs/serving-parity-2026-10
    # 5b. second-generation WebP loss on the sampled training crops (already WebP), per production model
    .venv/bin/python analysis/serving_parity.py webp-sample --sample-csv round3/sample.csv --crops round3/crops \\
        --api-dir ... --device mps --out-csv ~/Data/serving-parity-2026-10/webp_gen2.csv

``--points`` is a CSV ``point_id,x_norm,y_norm,note`` (default: :data:`DEFAULT_POINTS`). Poses are
:data:`POSES` at beta 0.95. The Depth Anything code picks cuda, else mps, else cpu for its *input* by itself; both
workers hide mps while running the reference depth (:func:`mps_hidden`) so the two sides run it on the same device,
and the train worker runs it once more on mps as a device-sensitivity probe.

Extra packages beyond requirements.txt: flask and the rest of sidewalk-ai-api's requirements.txt in the API venv.
The pure helpers (:func:`compare_arrays`, :func:`crop_half_train`, :func:`crop_half_api`, :func:`vote_parity`, ...) need
only numpy and are unit-tested in tests/test_serving_parity.py.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: sidewalk-ai-api's label-type keys -> the training repo's LabelType directory names.
LABEL_TYPES = {"curbramp": "CurbRamp", "nocurbramp": "NoCurbRamp", "obstacle": "Obstacle",
               "surfaceproblem": "SurfaceProblem", "crosswalk": "Crosswalk"}
CLASSES = ("correct", "incorrect")
GATE = 0.92  # SidewalkWebpage's ai-validation-min-accuracy (calibration._isotonic_map.GATE; repeated so CI needs nothing)
TILT_BETA = 0.95
CROP_DEPTH_CONSTANT = 6100
WEBP_QUALITY = 95
DEPTH_CKPT_NAME = "depth_anything_v2_metric_vkitti_vitb.pth"
STORE_CITY = "parity"           # the shim pano store's city directory
STORE_PANO_ID = "parityPano01"  # and the pano's id in it

#: Label points, normalised pano coordinates. P01 is a real curb ramp in sidewalk-panorama-tools'
#: samples/sample_pano.jpg (13312 x 6656; px (1604, 3753), hand-picked in that repo's assets/make_banner.py); the rest
#: are synthetic: sidewalk-level points round the view, the horizon, both sides of the seam, and near both poles.
DEFAULT_POINTS = [
    ("P01", 1604 / 13312, 3753 / 6656, "curb ramp (real feature)"),
    ("P02", 0.25, 0.60, "sidewalk level"),
    ("P03", 0.50, 0.60, "sidewalk level"),
    ("P04", 0.75, 0.62, "sidewalk level"),
    ("P05", 0.40, 0.70, "steep, near the rig"),
    ("P06", 0.10, 0.50, "horizon"),
    ("P07", 0.001, 0.60, "seam, left edge"),
    ("P08", 0.999, 0.58, "seam, right edge"),
    ("P09", 0.60, 0.92, "near nadir"),
    ("P10", 0.85, 0.08, "near zenith"),
]

#: pose_id -> (camera_pitch, camera_roll), degrees; None = unknown (the API gets no field).
POSES = {"level": (None, None), "pitch3": (3.0, None), "pitch-5_roll1": (-5.0, 1.0), "roll1_only": (None, 1.0)}
#: The pose the two sides deliberately read differently: training tilts by a roll with no pitch, the API does not.
ROLL_ONLY = "roll1_only"


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers (numpy only; unit-tested on CPU)
# ---------------------------------------------------------------------------------------------------------------------

def compare_arrays(a, b, centre=64):
    """How two images (or any same-shape arrays) differ: a dict of plain floats/ints.

    ``max_abs`` and ``mean_abs`` over all values, ``frac_diff`` = fraction of values that differ at all,
    ``frac_px_diff`` = fraction of pixels (last axis taken as channels when 3-D) with any channel different,
    ``centre_max_abs`` over the central ``centre`` x ``centre`` block, and ``identical`` (same shape, dtype and bytes).
    Shapes that differ give ``identical`` False and NaN for the rest.

    Example::

        >>> r = compare_arrays(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8), centre=2)
        >>> r["max_abs"], r["frac_px_diff"], r["identical"]
        (1.0, 1.0, False)
    """
    a, b = np.asarray(a), np.asarray(b)
    out = {"identical": bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())}
    if a.shape != b.shape:
        return {**out, "max_abs": float("nan"), "mean_abs": float("nan"), "frac_diff": float("nan"),
                "frac_px_diff": float("nan"), "centre_max_abs": float("nan"), "shape_a": a.shape, "shape_b": b.shape}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    px = d.max(axis=-1) if d.ndim == 3 else d
    h, w = d.shape[:2]
    c = d[max(0, h // 2 - centre // 2):h // 2 + centre // 2, max(0, w // 2 - centre // 2):w // 2 + centre // 2]
    return {**out, "max_abs": float(d.max()) if d.size else 0.0, "mean_abs": float(d.mean()) if d.size else 0.0,
            "frac_diff": float((d > 0).mean()) if d.size else 0.0,
            "frac_px_diff": float((px > 0).mean()) if px.size else 0.0,
            "centre_max_abs": float(c.max()) if c.size else 0.0}


def crop_half_train(depth_m):
    """``data_generation.depth_sized_crop``'s rule: ``int(6100 / depth)`` in float64 (``depth_m = float(depth[c, c])``).

    >>> crop_half_train(np.float32(10.0))
    610
    """
    return int(CROP_DEPTH_CONSTANT / float(depth_m))


def crop_half_api(depth_m):
    """sidewalk-ai-api ``main.py``'s rule, in its precision: ``inv_depth = 1 / depth[cy, cx]`` on the float32 depth map,
    then ``int(inv_depth * 6100)``. Both steps stay float32 (a numpy scalar with a Python number), so this can land one
    pixel off :func:`crop_half_train` when ``6100 / depth`` is within float32 rounding of an integer.

    >>> crop_half_api(np.float32(10.0))
    610
    """
    inv_depth = 1 / np.float32(depth_m)
    return int(inv_depth * CROP_DEPTH_CONSTANT)


def crop_half_disagreements(depths):
    """For an array of float32 depths: ``(n_disagree, indices)`` where the two crop-half rules give different ints."""
    d = np.asarray(depths, dtype=np.float32)
    bad = [i for i, v in enumerate(d) if crop_half_train(v) != crop_half_api(v)]
    return len(bad), bad


def crop_half_sweep(lo=1.0, hi=80.0):
    """Every float32 depth in ``[lo, hi)``, through both crop-half rules (:func:`crop_half_train` in float64,
    :func:`crop_half_api` in float32), vectorised. Returns ``(depths, mismatch)``: the float32 values and a boolean
    array marking where the two ints differ. Exhaustive, so it is the exact disagreement set, not a sample.

    >>> d, bad = crop_half_sweep(10.0, 10.001)
    >>> len(d) > 0, bool(bad.any())
    (True, False)
    """
    a = np.array([lo], np.float32).view(np.uint32)[0]
    b = np.array([hi], np.float32).view(np.uint32)[0]
    depths = np.arange(a, b, dtype=np.uint32).view(np.float32)
    train = np.floor(CROP_DEPTH_CONSTANT / depths.astype(np.float64))
    api = np.floor((np.float32(1) / depths) * np.float32(CROP_DEPTH_CONSTANT))
    return depths, train != api


def softmax32(logits):
    """Row-wise softmax in float32 (the API's ``F.softmax`` on float32 logits), numerically stable."""
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def api_lookup(class_map, confidence):
    """The API's accuracy lookup, copied from ``main.py`` (the same loop as ``calibration._isotonic_map.api_lookup``):
    the value of the last threshold, in insertion order, that is ``<= confidence``; 0 if none."""
    accuracy = 0
    for threshold, acc in class_map.items():
        if float(threshold) <= confidence:
            accuracy = acc
    return accuracy


def served_vote(probs, accuracy_map, gate=GATE):
    """``(verdict, confidence, served_accuracy, votes)`` for one crop's ``[p_correct, p_incorrect]``, by the API's rule.

    The verdict is ``argmax`` (first index on a tie, so ``correct``), the confidence is that class's probability, the
    served accuracy is :func:`api_lookup` on that class's map, and the verdict becomes a vote when it is ``>= gate``.

    >>> amap = {"correct": {"0.5": 0.5, "0.9": 0.95}, "incorrect": {"0.5": 0.4}}
    >>> served_vote([0.93, 0.07], amap)
    ('correct', 0.93, 0.95, True)
    """
    p = [float(v) for v in probs]
    k = int(np.argmax(p))
    verdict = CLASSES[k]
    acc = api_lookup(accuracy_map[verdict], p[k])
    return verdict, p[k], float(acc), bool(acc >= gate)


def vote_parity(ref_probs, probs, accuracy_map, gate=GATE):
    """Compare two pipelines' probabilities for the same crops: a dict of counts and maxima.

    ``verdict_flips``: argmax differs. ``vote_changes``: the served vote differs, i.e. one side votes and the other
    does not, or both vote with different verdicts (the rule of ``export_onnx.parity`` in the onnx-export branch).
    ``max_abs_dprob`` is over both classes.

    >>> amap = {"correct": {"0.5": 0.5, "0.9": 0.95}, "incorrect": {"0.5": 0.95}}
    >>> r = vote_parity([[0.91, 0.09], [0.6, 0.4]], [[0.89, 0.11], [0.6, 0.4]], amap)
    >>> r["vote_changes"], r["verdict_flips"], round(r["max_abs_dprob"], 2)
    (1, 0, 0.02)
    """
    ref = np.asarray(ref_probs, dtype=np.float64).reshape(-1, 2)
    new = np.asarray(probs, dtype=np.float64).reshape(-1, 2)
    if ref.shape != new.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {new.shape}")
    flips = changes = votes_ref = votes_new = 0
    for r, n in zip(ref, new):
        vr, _, _, okr = served_vote(r, accuracy_map, gate)
        vn, _, _, okn = served_vote(n, accuracy_map, gate)
        flips += vr != vn
        changes += (okr != okn) or (vr != vn and (okr or okn))
        votes_ref += okr
        votes_new += okn
    return {"n": int(len(ref)), "max_abs_dprob": float(np.abs(ref - new).max()) if len(ref) else 0.0,
            "verdict_flips": int(flips), "vote_changes": int(changes), "votes_ref": int(votes_ref),
            "votes": int(votes_new)}


def load_points(path=None):
    """``[(point_id, x_norm, y_norm, note), ...]`` from a CSV ``point_id,x_norm,y_norm[,note]``, or the defaults."""
    if path is None:
        return list(DEFAULT_POINTS)
    import csv
    with open(path, newline="") as f:
        return [(r["point_id"], float(r["x_norm"]), float(r["y_norm"]), r.get("note", "")) for r in csv.DictReader(f)]


def case_id(point_id, pose_id):
    """Directory name for one point x pose: ``P01__pitch3``."""
    return f"{point_id}__{pose_id}"


def sha256_file(path, chunk=1 << 20):
    """Hex sha256 of a file's bytes, read in ``chunk``-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a):
    """sha256 of an array's shape, dtype and bytes (so two arrays hash equal only when byte-identical)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.shape}|{a.dtype}|".encode() + a.tobytes()).hexdigest()


def load_model_map(api_dir):
    """sidewalk-ai-api's ``MODEL_MAP`` (a plain dict in ``sidewalk_ai_api/model_map.py``), without importing the API."""
    ns = {}
    with open(os.path.join(api_dir, "sidewalk_ai_api", "model_map.py")) as f:
        exec(compile(f.read(), "model_map.py", "exec"), ns)  # noqa: S102 - a read-only checkout we control
    return ns["MODEL_MAP"]


def load_accuracy_maps(api_dir):
    """``{api_type: {"correct": {...}, "incorrect": {...}}}`` from ``<api_dir>/accuracy_mappings``, key order kept."""
    maps = {}
    for t in LABEL_TYPES:
        with open(os.path.join(api_dir, "accuracy_mappings", f"{t}.json")) as f:
            maps[t] = json.load(f)
    return maps


def versions():
    """Versions of the packages that can change a pixel, for the provenance record."""
    out = {"python": platform.python_version(), "machine": f"{platform.system()} {platform.machine()}"}
    for name in ("numpy", "cv2", "PIL", "torch", "torchvision", "transformers", "sidewalk_panorama_tools"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except ImportError:
            out[name] = None
    try:
        from PIL import features
        out["libwebp"] = features.version("webp")
    except Exception:  # pragma: no cover - informational only
        out["libwebp"] = None
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Shared worker plumbing (needs torch)
# ---------------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def mps_hidden(hide=True):
    """Make ``torch.backends.mps.is_available()`` return False inside the block.

    The vendored Depth Anything ``image2tensor`` sends its input to cuda, else mps, else cpu, regardless of where the
    model is; on a Mac that is mps. Hiding mps puts the input on the cpu, next to a cpu model, as on both servers'
    CUDA path the input follows the model. (Without this the API's ``DepthAnythingPredictor``, which puts the model on
    "cuda else cpu", fails on a Mac with a device mismatch; see the doc.)
    """
    import torch
    original = torch.backends.mps.is_available
    if hide:
        torch.backends.mps.is_available = lambda: False
    try:
        yield
    finally:
        torch.backends.mps.is_available = original


def make_store(root, pano):
    """A one-pano pano store (``.pano-store`` marker + ``<city>/<id[:2]>/<id>.jpg`` symlink): what both sides read."""
    d = os.path.join(root, STORE_CITY, STORE_PANO_ID[:2])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, ".pano-store"), "w") as f:
        f.write("serving-parity shim store\n")
    link = os.path.join(d, f"{STORE_PANO_ID}.jpg")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.abspath(pano), link)
    return root


def make_run_dir(root, depth_checkpoint, extra=None):
    """A working directory holding what the two depth loaders read by relative path: ``Depth-Anything-V2`` (this
    repo's submodule) and the checkpoint, both as ``./<name>`` (training) and ``./checkpoints/<name>`` (the API)."""
    os.makedirs(os.path.join(root, "checkpoints"), exist_ok=True)
    links = {"Depth-Anything-V2": os.path.join(REPO_ROOT, "Depth-Anything-V2"),
             DEPTH_CKPT_NAME: os.path.abspath(depth_checkpoint),
             os.path.join("checkpoints", DEPTH_CKPT_NAME): os.path.abspath(depth_checkpoint), **(extra or {})}
    for name, target in links.items():
        p = os.path.join(root, name)
        if os.path.lexists(p):
            os.remove(p)
        os.symlink(target, p)
    return root


def save_png(rgb, path):
    """Lossless RGB PNG (PIL), so a crop moves between interpreters byte for byte."""
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(rgb)).save(path, format="PNG")


def load_png(path):
    """An image file as an (H, W, 3) uint8 RGB array (PIL, ``convert("RGB")``)."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def webp_roundtrip(rgb, path):
    """``data_generation.save_webp`` then decode as ``evaluate.run_inference`` does (``Image.open(...).convert("RGB")``)."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from data_generation import save_webp
    from PIL import Image
    save_webp(rgb, path)
    return np.asarray(Image.open(path).convert("RGB"))


def score(processor, model, rgb):
    """``(pixel_values, logits, probs)`` for one RGB crop through ``processor`` and ``model`` on the cpu, as
    ``evaluate.run_inference`` / the API's ``ImageValidator.validate`` do (batch 1, softmax over the logits)."""
    import torch
    from PIL import Image
    inputs = processor(Image.fromarray(np.ascontiguousarray(rgb)), return_tensors="pt")
    with torch.no_grad():
        logits = model(pixel_values=inputs["pixel_values"]).logits
    return (inputs["pixel_values"][0].numpy(), logits[0].numpy().astype(np.float64).tolist(),
            logits.softmax(dim=-1)[0].numpy().astype(np.float64).tolist())


def _write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------------------------------------------------------------
# Training-side worker
# ---------------------------------------------------------------------------------------------------------------------

class RecordingDepth:
    """Wraps a Depth Anything model: forwards ``infer_image`` and keeps its input and output."""

    def __init__(self, model):
        self.model, self.inputs, self.outputs = model, [], []

    def infer_image(self, image, *args, **kwargs):
        self.inputs.append(np.array(image))
        out = self.model.infer_image(image, *args, **kwargs)
        self.outputs.append(np.array(out))
        return out


def train_worker(args):
    """Stages 1-5 (+ the 4.55 half of 6) with the training code, writing ``<out>/index.json`` and per-case files."""
    import torch
    sys.path.insert(0, REPO_ROOT)
    import data_generation as dg

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    store = make_store(os.path.join(out, "store"), args.pano)
    run_dir = make_run_dir(os.path.join(out, "run"), args.depth_checkpoint)
    points = load_points(args.points)
    mps = bool(args.mps and torch.backends.mps.is_available())
    index = {"side": "train", "versions": versions(), "pano": os.path.abspath(args.pano),
             "pano_sha256": sha256_file(args.pano), "depth_checkpoint_sha256": sha256_file(args.depth_checkpoint),
             "mps": mps, "cases": {}}

    # Stage 1: read + resize, the training call.
    pano_bgr, info = dg.read_store_panorama(store, STORE_CITY, STORE_PANO_ID)
    if isinstance(pano_bgr, str):
        sys.exit(f"read_store_panorama: {pano_bgr}")
    np.save(os.path.join(out, "pano_bgr.npy"), pano_bgr)
    index.update(pano_native=[info["pano_native_w"], info["pano_native_h"]], pano_resized_sha256=sha256_array(pano_bgr))
    pano_rgb = np.ascontiguousarray(pano_bgr[..., ::-1])  # data_generation.main does exactly this

    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        with mps_hidden():
            depth_cpu = dg.load_depth_model("vitb", "cpu")
        depth_mps = dg.load_depth_model("vitb", "mps") if mps else None
    finally:
        os.chdir(cwd)

    for pid, x, y, note in points:
        for pose_id, (pitch, roll) in POSES.items():
            cid = case_id(pid, pose_id)
            cdir = os.path.join(out, cid)
            os.makedirs(cdir, exist_ok=True)
            rec = {"point_id": pid, "pose": pose_id, "x_norm": x, "y_norm": y, "note": note,
                   "camera_pitch": pitch, "camera_roll": roll}
            # Stage 2: the three training renderers.
            view_np, tilt_np = dg.render_label_view(pano_rgb, x, y, pitch, roll, beta=TILT_BETA, device=None)
            view_cpu, tilt_cpu = dg.render_label_view(pano_rgb, x, y, pitch, roll, beta=TILT_BETA, device="cpu")
            rec["tilt_applied"] = bool(tilt_cpu)
            rec["render_numpy_vs_torch_cpu"] = compare_arrays(view_np, view_cpu)
            view_mps = None
            if mps:
                try:  # torch 2.7's mps grid_sample has no padding_mode="border"; 2.14's has
                    view_mps, _ = dg.render_label_view(pano_rgb, x, y, pitch, roll, beta=TILT_BETA, device="mps")
                    rec["render_torch_cpu_vs_mps"] = compare_arrays(view_cpu, view_mps)
                except RuntimeError as e:
                    rec["render_mps_error"] = str(e)[:200]
            np.save(os.path.join(cdir, "view_rgb.npy"), view_cpu)
            np.save(os.path.join(cdir, "view_numpy.npy"), view_np)

            # Stages 3-4: the training composition itself, with the depth model recorded.
            rd = RecordingDepth(depth_cpu)
            with mps_hidden():
                crop, cinfo = dg.create_perspective_crop(rd, pano_rgb, x, y, camera_pitch=pitch, camera_roll=roll,
                                                         apply_tilt=True, beta=TILT_BETA, device="cpu")
            depth = rd.outputs[0]
            rec["depth_input_is_bgr_of_view"] = bool(np.array_equal(rd.inputs[0], view_cpu[..., ::-1]))
            c = dg.VIEW_SIZE // 2
            rec.update(depth_centre=float(depth[c, c]), crop_half_px=int(cinfo["crop_half_px"]),
                       clipped=bool(cinfo["clipped"]), crop_shape=list(crop.shape),
                       crop_half_api_rule=crop_half_api(depth[c, c]))
            np.save(os.path.join(cdir, "depth.npy"), depth)
            save_png(crop, os.path.join(cdir, "crop_raw.png"))
            # Probe crops for the model stage: the same depth on the numpy-rendered view (the --allow-cpu path) and on
            # the mps-rendered view (device rounding), and a crop one pixel larger (the crop-half rounding rule).
            depth_m = float(depth[c, c])
            save_png(dg.depth_sized_crop(view_np, depth_m)[0], os.path.join(cdir, "crop_numpy_render.png"))
            if view_mps is not None:
                save_png(dg.depth_sized_crop(view_mps, depth_m)[0], os.path.join(cdir, "crop_mps_render.png"))
            plus1, half1, _ = dg.depth_sized_crop(view_cpu, CROP_DEPTH_CONSTANT / (rec["crop_half_px"] + 1.5))
            assert half1 == rec["crop_half_px"] + 1, (half1, rec["crop_half_px"])
            save_png(plus1, os.path.join(cdir, "crop_half_plus1.png"))
            # Probe: the same view with its channels swapped (RGB into a BGR-expecting model).
            with mps_hidden():
                swapped = depth_cpu.infer_image(np.ascontiguousarray(view_cpu))
            rec["probe_swapped_depth_centre"] = float(swapped[c, c])
            rec["probe_swapped_crop_half_px"] = crop_half_train(swapped[c, c])
            if mps:
                try:
                    dm = depth_mps.infer_image(np.ascontiguousarray(view_cpu[..., ::-1]))
                    rec["probe_mps_depth_centre"] = float(dm[c, c])
                    rec["probe_mps_crop_half_px"] = crop_half_train(dm[c, c])
                    rec["probe_mps_depth_map_max_abs"] = float(np.abs(dm - depth).max())
                except RuntimeError as e:
                    rec["depth_mps_error"] = str(e)[:200]

            # Stage 5: WebP q95 (what training sees) and a second generation.
            webp1 = webp_roundtrip(crop, os.path.join(cdir, "crop.webp"))
            webp2 = webp_roundtrip(webp1, os.path.join(cdir, "crop_gen2.webp"))
            save_png(webp1, os.path.join(cdir, "crop_webp.png"))
            rec["webp_vs_raw"] = compare_arrays(crop, webp1)
            rec["webp_gen2_vs_gen1"] = compare_arrays(webp1, webp2)
            index["cases"][cid] = rec
            print(cid, "depth", round(rec["depth_centre"], 3), "half", rec["crop_half_px"], flush=True)

    if args.models:
        index["models"] = train_models(args, out, index)
    _write_json(index, os.path.join(out, "index.json"))


def train_models(args, out, index):
    """The training/eval preprocessing (``use_fast=False``) and each production validator on raw, WebP and
    second-generation WebP crops; the fast processor as a control. Saves the slow processor's pixel_values of the raw
    crop for the stage-6 comparison."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model_map = load_model_map(args.api_dir)["validator_models"]
    res = {}
    for t in LABEL_TYPES:
        entry = model_map[t]
        repo = f"projectsidewalk/{entry['model_id']}"
        slow = AutoImageProcessor.from_pretrained(repo, revision=entry["revision"], use_fast=False)
        fast = AutoImageProcessor.from_pretrained(repo, revision=entry["revision"], use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(repo, revision=entry["revision"]).eval()
        res[t] = {"repo": repo, "revision": entry["revision"], "processor": type(slow).__name__,
                  "fast_processor": type(fast).__name__, "id2label": model.config.id2label, "cases": {}}
        torch.manual_seed(0)
        for cid in index["cases"]:
            cdir = os.path.join(out, cid)
            raw, w1 = load_png(os.path.join(cdir, "crop_raw.png")), load_png(os.path.join(cdir, "crop_webp.png"))
            w2 = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(
                os.path.join(cdir, "crop_gen2.webp")).convert("RGB"))
            pv_raw, lg_raw, pr_raw = score(slow, model, raw)
            _, lg_w1, pr_w1 = score(slow, model, w1)
            _, lg_w2, pr_w2 = score(slow, model, w2)
            pv_fast = fast(__import__("PIL.Image", fromlist=["Image"]).fromarray(raw),
                           return_tensors="pt")["pixel_values"][0].numpy()
            np.save(os.path.join(cdir, f"pv_slow_{t}.npy"), pv_raw)
            rec = {"logits_raw": lg_raw, "probs_raw": pr_raw, "logits_webp": lg_w1, "probs_webp": pr_w1,
                   "logits_webp2": lg_w2, "probs_webp2": pr_w2,
                   "fast_vs_slow_pv_max_abs": float(np.abs(pv_fast - pv_raw).max())}
            for probe in ("numpy_render", "mps_render", "half_plus1"):
                f = os.path.join(cdir, f"crop_{probe}.png")
                if os.path.exists(f):
                    _, rec[f"logits_{probe}"], rec[f"probs_{probe}"] = score(slow, model, load_png(f))
            res[t]["cases"][cid] = rec
        print("scored", t, flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Serving-side worker (runs in the API's venv, against its checkout)
# ---------------------------------------------------------------------------------------------------------------------

def api_worker(args):
    """Drive sidewalk-ai-api's real ``/process`` for every point x pose and record each stage's array.

    Two things are not the API's own: the ``Depth-Anything-V2`` directory is this repo's submodule (the same commit,
    a561b84, as the API's, whose submodule is not initialised in a read-only checkout), and ``crop_half_px`` in the
    index is re-derived from the recorded depth map with the API's expression (:func:`crop_half_api`); the API does not
    return it. The crop itself is the PIL image ``/process`` hands the validator, recorded.
    """
    import torch
    api_dir = os.path.abspath(args.api_dir)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    train_dir = os.path.abspath(args.train_dir)
    with open(os.path.join(train_dir, "index.json")) as f:
        train_index = json.load(f)
    store = os.path.join(train_dir, "store")  # the same shim store the train worker read
    run_dir = make_run_dir(os.path.join(out, "run"), args.depth_checkpoint,
                           extra={"accuracy_mappings": os.path.join(api_dir, "accuracy_mappings"),
                                  "API_VERSION": os.path.join(api_dir, "API_VERSION")})
    os.environ["SIDEWALK_AI_API_KEY"] = "serving-parity"
    os.environ["SIDEWALK_SCRAPES_DIR"] = store
    os.chdir(run_dir)  # main.py and DepthAnythingPredictor read accuracy_mappings/, checkpoints/ by relative path
    sys.path.insert(0, api_dir)
    import main
    from sidewalk_ai_api import panorama as api_panorama
    from sidewalk_ai_api.depthanything import DepthAnythingPredictor

    index = {"side": "api", "versions": versions(), "api_dir": api_dir, "cases": {},
             "api_head": subprocess.run(["git", "-C", api_dir, "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()}

    # Probe: the API's depth loader as written, on this machine (mps visible).
    try:
        probe = DepthAnythingPredictor()
        probe.predict_depth(np.zeros((64, 64, 3), np.uint8))
        index["probe_depth_with_mps_visible"] = "ok"
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        index["probe_depth_with_mps_visible"] = f"{type(e).__name__}: {str(e)[:200]}"

    with mps_hidden():
        real_depth = DepthAnythingPredictor()

    class Recorder:
        depth_in, depth_out, crops, panos = [], [], [], []

    class RecDepth:
        def predict_depth(self, image):
            Recorder.depth_in.append(np.array(image))
            with mps_hidden():
                d = real_depth.predict_depth(image)
            Recorder.depth_out.append(np.array(d))
            return d

    class RecValidator:
        def validate(self, image):
            Recorder.crops.append(image.copy())
            return "correct", 0.5

    class StubTagger:
        def inference(self, image):
            return [], [], {}

    real_panorama = main.Panorama

    feed = np.load(args.pano_npy) if args.pano_npy else None
    index["pano_npy"] = os.path.abspath(args.pano_npy) if args.pano_npy else None

    def rec_panorama(pano_id, cached_image_path=None):
        p = real_panorama(pano_id, cached_image_path=cached_image_path)
        if feed is not None:  # cross-feed: everything after the read runs on this pano instead (stage-1 isolation)
            p.panorama_image = feed
        Recorder.panos.append((cached_image_path, p.image_source, p.panorama_image))
        return p

    main.Panorama = rec_panorama
    models = main.Models(depth=RecDepth(), taggers={t: StubTagger() for t in main.TAGGER_LABEL_TYPES},
                         validators={t: RecValidator() for t in main.VALIDATOR_LABEL_TYPES})
    client = main.create_app(models).test_client()
    headers = {"Authorization": "Bearer serving-parity"}
    saved_pano = False
    for cid, trec in train_index["cases"].items():
        data = {"label_type": "curbramp", "panorama_id": STORE_PANO_ID, "city": STORE_CITY,
                "x": repr(float(trec["x_norm"])), "y": repr(float(trec["y_norm"])), "tilt_beta": repr(TILT_BETA)}
        if trec["camera_pitch"] is not None:
            data["camera_pitch"] = repr(float(trec["camera_pitch"]))
        if trec["camera_roll"] is not None:
            data["camera_roll"] = repr(float(trec["camera_roll"]))
        for lst in (Recorder.depth_in, Recorder.depth_out, Recorder.crops, Recorder.panos):
            lst.clear()
        resp = client.post("/process", data=data, headers=headers)
        body = resp.get_json()
        if resp.status_code != 200:
            sys.exit(f"{cid}: /process {resp.status_code} {body}")
        cdir = os.path.join(out, cid)
        os.makedirs(cdir, exist_ok=True)
        path, source, pano = Recorder.panos[0]
        if not saved_pano:
            np.save(os.path.join(out, "pano_bgr.npy"), pano)
            index.update(pano_source=source, pano_cache_path_used=bool(path), pano_resized_sha256=sha256_array(pano))
            saved_pano = True
        view_bgr, depth = Recorder.depth_in[0], Recorder.depth_out[0]
        cy, cx = view_bgr.shape[0] // 2, view_bgr.shape[1] // 2
        crop_rgb = np.asarray(Recorder.crops[0])
        np.save(os.path.join(cdir, "view_bgr.npy"), view_bgr)
        np.save(os.path.join(cdir, "depth.npy"), depth)
        save_png(crop_rgb, os.path.join(cdir, "crop_raw.png"))
        index["cases"][cid] = {"response": body, "depth_centre": float(depth[cy, cx]),
                               "crop_half_px": crop_half_api(depth[cy, cx]), "crop_shape": list(crop_rgb.shape),
                               "crop_mode": Recorder.crops[0].mode}
        print(cid, "depth", round(float(depth[cy, cx]), 3), "tilt", body["tilt_applied"], flush=True)

    if args.models:
        index["models"] = api_models(args, out, train_dir, train_index)
    _write_json(index, os.path.join(out, "index.json"))


def api_models(args, out, train_dir, train_index):
    """The API's ``ImageValidator`` (``backend="pil"``) on the API's own crops (stage 7) and on the training worker's
    raw and WebP crops (stage 6 and the model-only part of 7), plus the default (torchvision) processor as a control."""
    from PIL import Image
    from transformers import AutoImageProcessor
    from sidewalk_ai_api.validator import ImageValidator
    res = {}
    for t in LABEL_TYPES:
        v = ImageValidator(label_type=t)
        v.model.eval()
        entry = load_model_map(args.api_dir)["validator_models"][t]
        default = AutoImageProcessor.from_pretrained(f"projectsidewalk/{entry['model_id']}", revision=entry["revision"])
        res[t] = {"processor": type(v.image_processor).__name__, "default_processor": type(default).__name__,
                  "device": str(next(v.model.parameters()).device), "cases": {}}
        for cid in train_index["cases"]:
            api_crop = load_png(os.path.join(out, cid, "crop_raw.png"))
            t_raw = load_png(os.path.join(train_dir, cid, "crop_raw.png"))
            t_webp = load_png(os.path.join(train_dir, cid, "crop_webp.png"))
            _, lg, pr = score(v.image_processor, v.model, api_crop)
            verdict, conf = v.validate(Image.fromarray(api_crop))  # the API's own call, to check score() matches it
            pv_t_raw, lg_t_raw, pr_t_raw = score(v.image_processor, v.model, t_raw)
            _, lg_t_webp, pr_t_webp = score(v.image_processor, v.model, t_webp)
            pv_slow = np.load(os.path.join(train_dir, cid, f"pv_slow_{t}.npy"))
            pv_default = default(Image.fromarray(t_raw), return_tensors="pt")["pixel_values"][0].numpy()
            res[t]["cases"][cid] = {
                "logits_api": lg, "probs_api": pr, "validate_verdict": verdict, "validate_confidence": conf,
                "logits_train_raw": lg_t_raw, "probs_train_raw": pr_t_raw,
                "logits_train_webp": lg_t_webp, "probs_train_webp": pr_t_webp,
                "pv_pil_vs_train_slow_max_abs": float(np.abs(pv_t_raw - pv_slow).max()),
                "pv_pil_vs_train_slow_identical": bool(np.array_equal(pv_t_raw, pv_slow)),
                "pv_default_vs_pil_max_abs": float(np.abs(pv_default - pv_t_raw).max())}
        print("scored", t, flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# Compare: the tables
# ---------------------------------------------------------------------------------------------------------------------

def _logit_diff(a, b):
    """Largest absolute difference between two logit vectors, in float64."""
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def compare(args):
    """Join the worker outputs into ``crops.csv`` (stages 1-4, one row per point x pose), ``models.csv`` (stages 5-7,
    one row per point x pose x model x comparison) and ``summary.md``."""
    import pandas as pd
    with open(os.path.join(args.train, "index.json")) as f:
        tr = json.load(f)
    with open(os.path.join(args.api, "index.json")) as f:
        ap = json.load(f)
    others = []
    for path in args.train_other or []:
        with open(os.path.join(path, "index.json")) as f:
            others.append((path, json.load(f)))
    amaps = load_accuracy_maps(args.api_dir)

    t_pano = np.load(os.path.join(args.train, "pano_bgr.npy"), mmap_mode="r")
    a_pano = np.load(os.path.join(args.api, "pano_bgr.npy"), mmap_mode="r")
    pano_cmp = compare_arrays(t_pano, a_pano, centre=0)
    other_pano = {}
    for path, oi in others:
        other_pano[path] = compare_arrays(t_pano, np.load(os.path.join(path, "pano_bgr.npy"), mmap_mode="r"), centre=0)

    def probe_env(key):
        """The first environment (primary, then others) whose cases carry ``key``: ``(tag, index, path)``."""
        for tag, idx, path in [("primary", tr, args.train)] + [(os.path.basename(os.path.normpath(p)), oi, p)
                                                              for p, oi in others]:
            if any(key in c for c in idx["cases"].values()):
                return tag, idx, path
        return None, None, None
    mps_tag, mps_idx, _ = probe_env("render_torch_cpu_vs_mps")

    rows = []
    for cid, t in tr["cases"].items():
        tm_ = mps_idx["cases"][cid] if mps_idx else {}
        a = ap["cases"][cid]
        tview = np.load(os.path.join(args.train, cid, "view_rgb.npy"))
        aview = np.load(os.path.join(args.api, cid, "view_bgr.npy"))[..., ::-1]
        tdepth = np.load(os.path.join(args.train, cid, "depth.npy"))
        adepth = np.load(os.path.join(args.api, cid, "depth.npy"))
        tcrop = load_png(os.path.join(args.train, cid, "crop_raw.png"))
        acrop = load_png(os.path.join(args.api, cid, "crop_raw.png"))
        v_ta = compare_arrays(tview, aview)
        crop_cmp = compare_arrays(tcrop, acrop)
        # Sensitivity: the same comparisons with one side's channels left in the wrong order must fail loudly.
        swapped_view = compare_arrays(tview, aview[..., ::-1])
        swapped_crop = compare_arrays(tcrop, acrop[..., ::-1])
        row = {"case": cid, "point_id": t["point_id"], "pose": t["pose"], "x_norm": t["x_norm"], "y_norm": t["y_norm"],
               "tilt_applied_train": t["tilt_applied"], "tilt_applied_api": a["response"]["tilt_applied"],
               "pano_identical": pano_cmp["identical"],
               "view_np_vs_torch_max": t["render_numpy_vs_torch_cpu"]["max_abs"],
               "view_np_vs_torch_frac_px": t["render_numpy_vs_torch_cpu"]["frac_px_diff"],
               "view_np_vs_torch_centre_max": t["render_numpy_vs_torch_cpu"]["centre_max_abs"],
               "mps_env": mps_tag,
               "view_cpu_vs_mps_max": tm_.get("render_torch_cpu_vs_mps", {}).get("max_abs"),
               "view_cpu_vs_mps_frac_px": tm_.get("render_torch_cpu_vs_mps", {}).get("frac_px_diff"),
               "view_cpu_vs_mps_centre_max": tm_.get("render_torch_cpu_vs_mps", {}).get("centre_max_abs"),
               "view_train_vs_api_identical": v_ta["identical"], "view_train_vs_api_max": v_ta["max_abs"],
               "view_train_vs_api_frac_px": v_ta["frac_px_diff"],
               "view_train_vs_api_centre_max": v_ta["centre_max_abs"],
               "depth_input_bgr_train": t["depth_input_is_bgr_of_view"],
               "depth_train": t["depth_centre"], "depth_api": a["depth_centre"],
               "depth_identical": bool(np.array_equal(tdepth, adepth)),
               "depth_map_max_abs": float(np.abs(tdepth.astype(np.float64) - adepth).max()),
               "depth_swapped": t["probe_swapped_depth_centre"], "half_swapped": t["probe_swapped_crop_half_px"],
               "depth_mps_minus_cpu": (tm_["probe_mps_depth_centre"] - tm_["depth_centre"]
                                       if "probe_mps_depth_centre" in tm_ else None),
               "half_mps_changed": (tm_["probe_mps_crop_half_px"] != tm_["crop_half_px"]
                                    if "probe_mps_crop_half_px" in tm_ else None),
               "half_train": t["crop_half_px"], "half_api": a["crop_half_px"],
               "half_api_rule_on_train_depth": t["crop_half_api_rule"],
               "clipped_train": t["clipped"],
               "crop_identical": crop_cmp["identical"], "crop_max_abs": crop_cmp["max_abs"],
               "crop_mean_abs": crop_cmp["mean_abs"],
               "probe_view_bgr_vs_rgb_mean_abs": swapped_view["mean_abs"],
               "probe_crop_bgr_vs_rgb_mean_abs": swapped_crop["mean_abs"],
               "webp_max_abs": t["webp_vs_raw"]["max_abs"], "webp_mean_abs": t["webp_vs_raw"]["mean_abs"],
               "webp2_mean_abs": t["webp_gen2_vs_gen1"]["mean_abs"]}
        for path, oi in others:
            o = oi["cases"][cid]
            tag = os.path.basename(os.path.normpath(path))
            oview = np.load(os.path.join(path, cid, "view_rgb.npy"))
            ov = compare_arrays(tview, oview)
            row.update({f"{tag}_view_identical": ov["identical"], f"{tag}_view_max": ov["max_abs"],
                        f"{tag}_view_frac_px": ov["frac_px_diff"], f"{tag}_depth": o["depth_centre"],
                        f"{tag}_half": o["crop_half_px"],
                        f"{tag}_crop_identical": compare_arrays(tcrop, load_png(os.path.join(path, cid, "crop_raw.png")))
                        ["identical"],
                        f"{tag}_vs_api_view_identical": bool(np.array_equal(oview, aview)),
                        f"{tag}_vs_api_depth_identical": bool(np.array_equal(
                            np.load(os.path.join(path, cid, "depth.npy")), adepth)),
                        f"{tag}_vs_api_crop_identical": bool(np.array_equal(
                            load_png(os.path.join(path, cid, "crop_raw.png")), acrop))})
        rows.append(row)
    crops = pd.DataFrame(rows)

    mrows = []
    if "models" in tr and "models" in ap:
        for mt in LABEL_TYPES:
            amap = amaps[mt]
            for cid in tr["cases"]:
                tm, am = tr["models"][mt]["cases"][cid], ap["models"][mt]["cases"][cid]
                comps = {  # name: (reference logits/probs, other logits/probs)
                    "5_webp_vs_raw (train env)": (tm["logits_raw"], tm["probs_raw"], tm["logits_webp"], tm["probs_webp"]),
                    "5_webp2_vs_webp (train env)": (tm["logits_webp"], tm["probs_webp"], tm["logits_webp2"],
                                                    tm["probs_webp2"]),
                    "6+7_model_env (same raw crop: train env vs API env)": (tm["logits_raw"], tm["probs_raw"],
                                                                         am["logits_train_raw"], am["probs_train_raw"]),
                    "7_end_to_end (train webp in train env vs API)": (tm["logits_webp"], tm["probs_webp"],
                                                                      am["logits_api"], am["probs_api"]),
                    "7_raw_crops (train raw vs API raw, API env)": (am["logits_train_raw"], am["probs_train_raw"],
                                                                    am["logits_api"], am["probs_api"]),
                }
                for probe, label in (("numpy_render", "2_numpy_vs_torch_render (raw crop, train env)"),
                                     ("mps_render", "2_mps_vs_cpu_render (raw crop, train env)"),
                                     ("half_plus1", "4_crop_half_plus1 (raw crop, train env)")):
                    if f"logits_{probe}" in tm:
                        comps[label] = (tm["logits_raw"], tm["probs_raw"], tm[f"logits_{probe}"], tm[f"probs_{probe}"])
                for name, (lr, prr, ln, pn) in comps.items():
                    vp = vote_parity([prr], [pn], amap)
                    mrows.append({"model": mt, "case": cid, "pose": tr["cases"][cid]["pose"], "comparison": name,
                                  "max_abs_dlogit": _logit_diff(lr, ln),
                                  "max_abs_dprob": vp["max_abs_dprob"], "verdict_flip": vp["verdict_flips"],
                                  "vote_change": vp["vote_changes"], "vote_ref": vp["votes_ref"], "vote": vp["votes"],
                                  "p_incorrect_ref": prr[1]})
                mrows.append({"model": mt, "case": cid, "comparison": "6_pixel_values (4.55 slow vs API pil)",
                              "max_abs_dpixel": am["pv_pil_vs_train_slow_max_abs"],
                              "identical": am["pv_pil_vs_train_slow_identical"]})
                mrows.append({"model": mt, "case": cid, "comparison": "6_control (4.55 fast vs 4.55 slow)",
                              "max_abs_dpixel": tm["fast_vs_slow_pv_max_abs"]})
                mrows.append({"model": mt, "case": cid, "comparison": "6_control (API default backend vs pil)",
                              "max_abs_dpixel": am["pv_default_vs_pil_max_abs"]})
                mrows.append({"model": mt, "case": cid, "comparison": "api_validate_matches_score",
                              "identical": bool(am["validate_verdict"] == CLASSES[int(np.argmax(am["probs_api"]))]
                                                and abs(am["validate_confidence"] - max(am["probs_api"])) < 1e-7)})
    for path, oi in others:  # another training environment: its own crops through its own processor and models
        tag = os.path.basename(os.path.normpath(path))
        if "models" not in oi or "models" not in tr:
            continue
        for mt in LABEL_TYPES:
            for cid in tr["cases"]:
                tm, om = tr["models"][mt]["cases"][cid], oi["models"][mt]["cases"][cid]
                vp = vote_parity([tm["probs_webp"]], [om["probs_webp"]], amaps[mt])
                pv_a = np.load(os.path.join(args.train, cid, f"pv_slow_{mt}.npy"))
                pv_b = np.load(os.path.join(path, cid, f"pv_slow_{mt}.npy"))
                mrows.append({"model": mt, "case": cid, "pose": tr["cases"][cid]["pose"],
                              "comparison": f"env_{tag} (webp crop, own env end to end)",
                              "max_abs_dlogit": _logit_diff(tm["logits_webp"], om["logits_webp"]),
                              "max_abs_dprob": vp["max_abs_dprob"], "verdict_flip": vp["verdict_flips"],
                              "vote_change": vp["vote_changes"], "vote_ref": vp["votes_ref"], "vote": vp["votes"]})
                for probe, label in (("numpy_render", "2_numpy_vs_torch_render"), ("mps_render", "2_mps_vs_cpu_render"),
                                     ("half_plus1", "4_crop_half_plus1")):
                    if f"logits_{probe}" in om:
                        vq = vote_parity([om["probs_raw"]], [om[f"probs_{probe}"]], amaps[mt])
                        mrows.append({"model": mt, "case": cid, "pose": tr["cases"][cid]["pose"],
                                      "comparison": f"{label} (raw crop, {tag} env)",
                                      "max_abs_dlogit": _logit_diff(om["logits_raw"], om[f"logits_{probe}"]),
                                      "max_abs_dprob": vq["max_abs_dprob"], "verdict_flip": vq["verdict_flips"],
                                      "vote_change": vq["vote_changes"], "vote_ref": vq["votes_ref"],
                                      "vote": vq["votes"]})
                am = ap["models"][mt]["cases"][cid]
                va = vote_parity([om["probs_raw"]], [am["probs_api"]], amaps[mt])
                mrows.append({"model": mt, "case": cid, "pose": tr["cases"][cid]["pose"],
                              "comparison": f"7_raw_end_to_end ({tag} env vs API)",
                              "max_abs_dlogit": _logit_diff(om["logits_raw"], am["logits_api"]),
                              "max_abs_dprob": va["max_abs_dprob"], "verdict_flip": va["verdict_flips"],
                              "vote_change": va["vote_changes"], "vote_ref": va["votes_ref"], "vote": va["votes"]})
                mrows.append({"model": mt, "case": cid, "comparison": f"6_env_{tag} pixel_values (own raw crop)",
                              "max_abs_dpixel": float(np.abs(pv_a - pv_b).max()),
                              "identical": bool(np.array_equal(pv_a, pv_b))})
    models = pd.DataFrame(mrows)

    os.makedirs(args.out, exist_ok=True)
    crops.to_csv(os.path.join(args.out, "crops.csv"), index=False, float_format="%.6g")
    models.to_csv(os.path.join(args.out, "models.csv"), index=False, float_format="%.6g")
    summary = summarize(crops, models, tr, ap, pano_cmp, other_pano)
    for path in getattr(args, "append", None) or []:  # isolate / pixel-compare outputs, verbatim
        with open(path) as f:
            summary += f"\n## {os.path.basename(path)}\n\n```\n{f.read().rstrip()}\n```\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary)
    _write_json({"train_versions": tr["versions"], "api_versions": ap["versions"],
                 "other_train_versions": {os.path.basename(os.path.normpath(p)): oi["versions"] for p, oi in others},
                 "api_head": ap.get("api_head"), "pano_sha256": tr["pano_sha256"],
                 "pano_native": tr["pano_native"], "depth_checkpoint_sha256": tr["depth_checkpoint_sha256"],
                 "probe_api_depth_with_mps_visible": ap.get("probe_depth_with_mps_visible"),
                 "pano_train_vs_api": pano_cmp, "pano_train_vs_other": {os.path.basename(os.path.normpath(k)): v for k, v in other_pano.items()},
                 "api_pano_source": ap.get("pano_source")}, os.path.join(args.out, "environments.json"))
    print(summary)


def _md(df, floatfmt="{:.4g}"):
    """A small markdown table (no tabulate dependency)."""
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def summarize(crops, models, tr, ap, pano_cmp, other_pano):
    """The markdown summary: stages 1-4 over the poses both sides read the same way, the ``roll1_only`` pose on its own
    (a known frame-rule difference, not rounding), and stages 5-7 per model."""
    import pandas as pd
    s = ["# Serving parity summary\n",
         f"train env: {json.dumps(tr['versions'])}\n\napi env: {json.dumps(ap['versions'])}\n",
         f"{len(crops)} cases ({crops.point_id.nunique()} points x {crops.pose.nunique()} poses). "
         f"`{ROLL_ONLY}` (a roll with no pitch) is reported apart: the API ignores a roll without a pitch.\n"]

    def stage_rows(c, tag):
        n = len(c)
        if n == 0:
            return []
        return [
            (tag, "2 view train torch cpu vs API cpu", f"identical {int(c.view_train_vs_api_identical.sum())}/{n}, "
             f"max {c.view_train_vs_api_max.max():g}"),
            (tag, "2 tilt_applied agrees", f"{int((c.tilt_applied_train == c.tilt_applied_api).sum())}/{n}"),
            (tag, "3 depth map identical train vs API", f"{int(c.depth_identical.sum())}/{n}, max "
             f"{c.depth_map_max_abs.max():g} m"),
            (tag, "4 crop_half train vs API", f"equal {int((c.half_train == c.half_api).sum())}/{n}"),
            (tag, "4 crop bytes train vs API", f"identical {int(c.crop_identical.sum())}/{n}, max "
             f"{c.crop_max_abs.max():g}, mean abs max {c.crop_mean_abs.max():.3g}"),
        ]

    matched, roll = crops[crops.pose != ROLL_ONLY], crops[crops.pose == ROLL_ONLY]
    n = len(matched)
    st = [("all", "1 pano read+resize, train vs API", f"identical={pano_cmp['identical']}, max {pano_cmp['max_abs']:g}")]
    for p, c in other_pano.items():
        st.append(("all", f"1 pano, train vs {os.path.basename(os.path.normpath(p))}",
                   f"identical={c['identical']}, max {c['max_abs']:g}, frac {c['frac_diff']:.2e}"))
    st += stage_rows(matched, "matched poses") + stage_rows(roll, ROLL_ONLY)
    for col in [c for c in crops.columns if c.endswith("_vs_api_view_identical")]:
        tag = col[:-len("_vs_api_view_identical")]
        for name, c in (("matched poses", matched), (ROLL_ONLY, roll)):
            if len(c):
                st.append((name, f"2-4 {tag} env vs API: view / depth map / crop identical",
                           f"{int(c[col].sum())} / {int(c[tag + '_vs_api_depth_identical'].sum())} / "
                           f"{int(c[tag + '_vs_api_crop_identical'].sum())} of {len(c)}"))
    st += [
        ("all", "2 view numpy vs torch cpu (train)", f"max {crops.view_np_vs_torch_max.max():g}, frac px median "
         f"{crops.view_np_vs_torch_frac_px.median():.2e} max {crops.view_np_vs_torch_frac_px.max():.2e}, centre max "
         f"{crops.view_np_vs_torch_centre_max.max():g}"),
        ("all", f"2 view torch cpu vs mps ({crops.mps_env.iloc[0]} env)", f"max {crops.view_cpu_vs_mps_max.max():g}, frac px max "
         f"{crops.view_cpu_vs_mps_frac_px.max():.2e}, centre max {crops.view_cpu_vs_mps_centre_max.max():g}"),
        ("all", "3 depth input is the BGR view (train)", f"{int(crops.depth_input_bgr_train.sum())}/{len(crops)}"),
        ("all", "3 probe: channels swapped", f"centre depth change median "
         f"{(crops.depth_swapped - crops.depth_train).abs().median():.3g} m, max "
         f"{(crops.depth_swapped - crops.depth_train).abs().max():.3g} m; crop_half changed "
         f"{int((crops.half_swapped != crops.half_train).sum())}/{len(crops)}"),
        ("all", "3 probe: depth on mps vs cpu", f"centre max {crops.depth_mps_minus_cpu.abs().max():.3g} m; "
         f"crop_half changed {int(crops.half_mps_changed.fillna(False).astype(bool).sum())}/{len(crops)} "
         f"({crops.mps_env.iloc[0]} env)"),
        ("all", "4 API crop-half rule on the training depth", f"equal "
         f"{int((crops.half_api_rule_on_train_depth == crops.half_train).sum())}/{len(crops)}"),
        ("matched poses", "probe: a channel-order bug would show as", f"view mean abs >= "
         f"{matched.probe_view_bgr_vs_rgb_mean_abs.min():.3g}, crop mean abs >= "
         f"{matched.probe_crop_bgr_vs_rgb_mean_abs.min():.3g} levels" if n else ""),
        ("all", "5 WebP q95 vs raw (pixels)", f"mean abs median {crops.webp_mean_abs.median():.3g}, max abs max "
         f"{crops.webp_max_abs.max():g}; gen2 vs gen1 mean abs median {crops.webp2_mean_abs.median():.3g}"),
    ]
    s.append(_md(pd.DataFrame(st, columns=["cases", "stage", "result"])))
    if len(models):
        lm = models[models.max_abs_dlogit.notna()].copy()
        lm["cases"] = np.where(lm.pose == ROLL_ONLY, ROLL_ONLY, "matched poses")
        g = lm.groupby(["comparison", "cases", "model"]).agg(
            n=("case", "size"), max_dlogit=("max_abs_dlogit", "max"), median_dprob=("max_abs_dprob", "median"),
            p90_dprob=("max_abs_dprob", lambda x: x.quantile(0.9)), max_dprob=("max_abs_dprob", "max"), verdict_flips=("verdict_flip", "sum"),
            vote_changes=("vote_change", "sum"), votes_ref=("vote_ref", "sum")).reset_index()
        pooled = lm.groupby(["comparison", "cases"]).agg(
            n=("case", "size"), max_dlogit=("max_abs_dlogit", "max"), median_dprob=("max_abs_dprob", "median"),
            p90_dprob=("max_abs_dprob", lambda x: x.quantile(0.9)), max_dprob=("max_abs_dprob", "max"),
            verdict_flips=("verdict_flip", "sum"), vote_changes=("vote_change", "sum"),
            votes_ref=("vote_ref", "sum")).reset_index()
        s.append("\n## Model outputs, five models pooled (stages 2, 4, 5-7)\n\n" + _md(pooled))
        s.append("\n## Model outputs per model (stages 2, 4, 5-7)\n\n" + _md(g))
        pm = models[models.comparison.str.startswith("6_")]
        g6 = pm.groupby(["comparison", "model"]).agg(n=("case", "size"),
                                                     max_dpixel=("max_abs_dpixel", "max")).reset_index()
        s.append("\n## Preprocessing (stage 6)\n\n" + _md(g6))
        ok = models[models.comparison == "api_validate_matches_score"]
        s.append(f"\nAPI `validate()` verdict/confidence reproduced by the recorder: {int(ok.identical.sum())}/{len(ok)}\n")
    return "\n".join(s)


# ---------------------------------------------------------------------------------------------------------------------
# 5b: second-generation WebP on the sampled training crops
# ---------------------------------------------------------------------------------------------------------------------

def webp_sample(args):
    """Score every sampled crop (already a q95 WebP) and the same crop re-encoded once more at q95, with its type's
    production model, slow processor; write per-label probabilities to ``--out-csv`` (outside the repo) and print a
    per-type summary. The second generation's loss stands in for the first's, which cannot be recomputed without the
    raw crop (the panoramas are gone); the real-pano cases measure both and give the ratio."""
    import io
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    sys.path.insert(0, REPO_ROOT)
    import evaluate
    df = pd.read_csv(args.sample_csv)
    if args.limit:
        df = df.groupby(["label_type", "split", "label_class"], group_keys=False).head(args.limit)
    model_map = load_model_map(args.api_dir)["validator_models"]
    amaps = load_accuracy_maps(args.api_dir)
    device = args.device
    out_rows = []
    for api_t, lt in LABEL_TYPES.items():
        sub = df[df.label_type == lt]
        if sub.empty:
            continue
        entry = model_map[api_t]
        repo = f"projectsidewalk/{entry['model_id']}"
        proc = AutoImageProcessor.from_pretrained(repo, revision=entry["revision"], use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(repo, revision=entry["revision"]).to(device).eval()
        for r in sub.itertuples():
            path = os.path.join(args.crops, lt, r.crop_path)
            g1 = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            g1.save(buf, format="WEBP", quality=WEBP_QUALITY)  # data_generation.save_webp's call, in memory
            g2 = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            px = proc([g1, g2], return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                p = model(pixel_values=px).logits.softmax(dim=-1).float().cpu().numpy()
            d = np.abs(np.asarray(g1, dtype=np.int16) - np.asarray(g2, dtype=np.int16))
            out_rows.append({"label_uid": r.label_uid, "label_type": lt, "split": r.split, "true": r.label_class,
                             "p_incorrect_gen1": float(p[0, 1]), "p_incorrect_gen2": float(p[1, 1]),
                             "p_correct_gen1": float(p[0, 0]), "p_correct_gen2": float(p[1, 0]),
                             "pixel_mean_abs": float(d.mean()), "pixel_max_abs": int(d.max())})
        print("done", lt, flush=True)
    res = pd.DataFrame(out_rows)
    if not res.label_uid.is_unique:
        raise ValueError("label_uid is not unique in the sample")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    res.to_csv(args.out_csv, index=False)
    summ = []
    for lt, g in res.groupby("label_type"):
        api_t = {v: k for k, v in LABEL_TYPES.items()}[lt]
        pr1 = g[["p_correct_gen1", "p_incorrect_gen1"]].values
        pr2 = g[["p_correct_gen2", "p_incorrect_gen2"]].values
        vp = vote_parity(pr1, pr2, amaps[api_t])
        dp = np.abs(pr1[:, 1] - pr2[:, 1])
        m1 = evaluate.slice_metrics(pd.DataFrame({"true": g.true, "p_correct": pr1[:, 0], "p_incorrect": pr1[:, 1]}))
        m2 = evaluate.slice_metrics(pd.DataFrame({"true": g.true, "p_correct": pr2[:, 0], "p_incorrect": pr2[:, 1]}))
        summ.append({"label_type": lt, "n": len(g), "median_dprob": float(np.median(dp)),
                     "p99_dprob": float(np.quantile(dp, 0.99)), "max_dprob": float(dp.max()),
                     "verdict_flips": vp["verdict_flips"], "vote_changes": vp["vote_changes"],
                     "votes_gen1": vp["votes_ref"], "votes_gen2": vp["votes"],
                     "auc_gen1": m1["auc"], "auc_gen2": m2["auc"],
                     "bal_acc_gen1": m1["balanced_accuracy"], "bal_acc_gen2": m2["balanced_accuracy"],
                     "pixel_mean_abs_median": float(g.pixel_mean_abs.median())})
    summ = pd.DataFrame(summ)
    if args.summary_csv:
        summ.to_csv(args.summary_csv, index=False, float_format="%.6g")
    print(_md(summ))
    print("per-label csv sha256", sha256_file(args.out_csv), "rows", len(res))


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def pixel_probe(args):
    """Per-interpreter pixel facts, for comparing package builds without any model: the panorama's JPEG decode, the
    8192 x 4096 ``INTER_LINEAR`` resize of that decode and of ``--reference-decode`` (another build's decode, so the
    resize is compared on identical input), ``INTER_CUBIC`` on a fixed crop of it, and the q95 WebP bytes PIL writes
    for every ``--crops`` PNG (``data_generation.save_webp``'s call, in memory). Writes ``<out>/probe.json`` + arrays."""
    import io
    import cv2
    from PIL import Image
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    dec = cv2.imread(args.pano)
    np.save(os.path.join(out, "decode.npy"), dec)
    ref = np.load(args.reference_decode) if args.reference_decode else dec
    rs = cv2.resize(ref, (8192, 4096), interpolation=cv2.INTER_LINEAR)
    np.save(os.path.join(out, "resize_linear.npy"), rs)
    region = ref[3000:3600, 1200:1800]
    cub = cv2.resize(region, (640, 640), interpolation=cv2.INTER_CUBIC)
    np.save(os.path.join(out, "resize_cubic.npy"), cub)
    webp = {}
    for path in sorted(args.crops or []):
        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, format="WEBP", quality=WEBP_QUALITY)
        webp[os.path.relpath(path, args.crops_root) if args.crops_root else path] = hashlib.sha256(
            buf.getvalue()).hexdigest()
    _write_json({"versions": versions(), "decode_sha": sha256_array(dec), "reference_decode": args.reference_decode,
                 "resize_linear_sha": sha256_array(rs), "resize_cubic_sha": sha256_array(cub), "webp_q95_sha": webp},
                os.path.join(out, "probe.json"))


def pixel_compare(args):
    """Compare two :func:`pixel_probe` outputs (``a`` should have been run with ``--reference-decode`` = ``b``'s
    decode, or vice versa); print one line per check."""
    pa, pb = (json.load(open(os.path.join(d, "probe.json"))) for d in (args.a, args.b))
    lines = []
    for name in ("decode", "resize_linear", "resize_cubic"):
        x, y = np.load(os.path.join(args.a, f"{name}.npy")), np.load(os.path.join(args.b, f"{name}.npy"))
        c = compare_arrays(x, y, centre=0)
        lines.append(f"{name}: identical={c['identical']} max_abs={c['max_abs']:g} frac_values={c['frac_diff']:.4f}")
    keys = sorted(set(pa["webp_q95_sha"]) & set(pb["webp_q95_sha"]))
    same = sum(pa["webp_q95_sha"][k] == pb["webp_q95_sha"][k] for k in keys)
    lines.append(f"webp q95 bytes identical: {same}/{len(keys)} crops "
                 f"(PIL {pa['versions']['PIL']} libwebp {pa['versions']['libwebp']} vs "
                 f"PIL {pb['versions']['PIL']} libwebp {pb['versions']['libwebp']})")
    lines.insert(0, f"a: {json.dumps(pa['versions'])}\nb: {json.dumps(pb['versions'])}")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text, end="")


def isolate(args):
    """Stage-1 isolation: the train worker's arrays against an api-worker run that was cross-fed the train worker's
    ``pano_bgr.npy`` (``api-worker --pano-npy``). If views, depth maps and crops are identical, every stage-2-4
    difference in the normal run comes from the panorama read; what remains otherwise belongs to the rest of the
    environment (torch, numpy). Prints a table; ``--out`` writes it."""
    import pandas as pd
    tr = json.load(open(os.path.join(args.train, "index.json")))
    ap = json.load(open(os.path.join(args.api, "index.json")))
    if not ap.get("pano_npy"):
        sys.exit(f"{args.api} was not a cross-fed run (api-worker --pano-npy)")
    rows = []
    for cid, t in tr["cases"].items():
        tview = np.load(os.path.join(args.train, cid, "view_rgb.npy"))
        aview = np.load(os.path.join(args.api, cid, "view_bgr.npy"))[..., ::-1]
        tdepth = np.load(os.path.join(args.train, cid, "depth.npy"))
        adepth = np.load(os.path.join(args.api, cid, "depth.npy"))
        vc = compare_arrays(tview, aview)
        cc = compare_arrays(load_png(os.path.join(args.train, cid, "crop_raw.png")),
                            load_png(os.path.join(args.api, cid, "crop_raw.png")))
        rows.append({"case": cid, "pose": t["pose"], "view_identical": vc["identical"], "view_max": vc["max_abs"],
                     "view_frac_px": vc["frac_px_diff"], "depth_identical": bool(np.array_equal(tdepth, adepth)),
                     "depth_max_abs": float(np.abs(tdepth.astype(np.float64) - adepth).max()),
                     "half_equal": t["crop_half_px"] == ap["cases"][cid]["crop_half_px"],
                     "crop_identical": cc["identical"], "crop_mean_abs": cc["mean_abs"]})
    df = pd.DataFrame(rows)
    lines = [f"train env: {json.dumps(tr['versions'])}", f"api env: {json.dumps(ap['versions'])}",
             f"api fed: {os.path.basename(ap['pano_npy'])} (the train worker's read+resize)"]
    for name, g in (("matched poses", df[df.pose != ROLL_ONLY]), (ROLL_ONLY, df[df.pose == ROLL_ONLY])):
        if len(g):
            lines.append(f"{name}: view identical {int(g.view_identical.sum())}/{len(g)} (max {g.view_max.max():g}, "
                         f"px frac max {g.view_frac_px.max():.2e}); depth identical {int(g.depth_identical.sum())}/"
                         f"{len(g)} (max {g.depth_max_abs.max():.3g} m); crop_half equal {int(g.half_equal.sum())}/"
                         f"{len(g)}; crop identical {int(g.crop_identical.sum())}/{len(g)} (mean abs max "
                         f"{g.crop_mean_abs.max():.3g})")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        df.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False, float_format="%.6g")
    print(text, end="")


def depth_margin_mm(depth_m):
    """How far, in millimetres, the centre depth may move before training's crop half-size ``int(6100 / d)`` changes:
    the distance from ``6100 / d`` to the nearest integer, times ``d**2 / 6100`` (the first-order depth change per unit
    of ``6100 / d``). A case with a small margin can cut a crop one pixel larger or smaller on another device.

    >>> round(depth_margin_mm(6100 / 100.5), 1)   # 6100 / d = 100.5: half a unit from either boundary
    302.0
    """
    q = CROP_DEPTH_CONSTANT / float(depth_m)
    frac = q - np.floor(q)
    return float(min(frac, 1 - frac) * float(depth_m) ** 2 / CROP_DEPTH_CONSTANT * 1000)


def expected_values(args):
    """The deploy smoke test's expected outputs: for each point x matched pose x model, what ``/process`` should return
    once it serves training's input (opencv 4.12 resize + q95 WebP round trip), i.e. the train worker's probabilities on
    its decoded WebP crop, beside what the API returns today (``--api``, its raw crop). ``roll1_only`` is left out:
    the two sides read that pose differently by design. ``in_smoke_set`` is False for cases whose centre depth is within
    ``--min-margin-mm`` of a crop-half boundary (:func:`depth_margin_mm`): a device-level depth change there flips the
    crop by a pixel, which moves p by up to 0.19, so those rows cannot be held to a 0.01 tolerance. Writes ``--out``."""
    import pandas as pd
    tr = json.load(open(os.path.join(args.train, "index.json")))
    ap = json.load(open(os.path.join(args.api, "index.json")))
    rows = []
    for cid, t in tr["cases"].items():
        if t["pose"] == ROLL_ONLY:
            continue
        for mt in LABEL_TYPES:
            pt = tr["models"][mt]["cases"][cid]["probs_webp"]
            pa = ap["models"][mt]["cases"][cid]["probs_api"]
            margin = depth_margin_mm(t["depth_centre"])
            rows.append({"point_id": t["point_id"], "pose": t["pose"], "x": repr(float(t["x_norm"])),
                         "y": repr(float(t["y_norm"])),
                         "camera_pitch": t["camera_pitch"], "camera_roll": t["camera_roll"], "tilt_beta": TILT_BETA,
                         "label_type": mt, "expected_crop_half_px": t["crop_half_px"],
                         "depth_margin_mm": round(margin, 3), "in_smoke_set": bool(margin >= args.min_margin_mm),
                         "expected_p_incorrect": round(pt[1], 6),
                         "expected_validation_result": CLASSES[int(np.argmax(pt))],
                         "api_174a169_p_incorrect": round(pa[1], 6),
                         "api_174a169_abs_diff": round(abs(pt[1] - pa[1]), 6)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    smoke, out = df[df.in_smoke_set], df[~df.in_smoke_set]
    print(f"{len(df)} rows; today's API is off by > 0.01 in {int((df.api_174a169_abs_diff > 0.01).sum())}, "
          f"max {df.api_174a169_abs_diff.max():.4f}. Smoke set (depth margin >= {args.min_margin_mm:g} mm): "
          f"{len(smoke)} rows, {smoke.groupby(['point_id', 'pose']).ngroups} cases; today's API misses "
          f"{int((smoke.api_174a169_abs_diff > 0.01).sum())}, max {smoke.api_174a169_abs_diff.max():.4f}. "
          f"Excluded cases: {sorted(set(out.point_id + '__' + out.pose))}")


def crop_half_report(args):
    """Stage 4b: how often the API's float32 crop-half arithmetic disagrees with training's float64, exhaustively over
    float32 depths in ``[--lo, --hi)``, and weighted by the depths in ``--sample-csv`` (its ``depth_m`` column)."""
    depths, bad = crop_half_sweep(args.lo, args.hi)
    diff = (np.floor((np.float32(1) / depths[bad]) * np.float32(CROP_DEPTH_CONSTANT))
            - np.floor(CROP_DEPTH_CONSTANT / depths[bad].astype(np.float64)))
    print(f"float32 depths in [{args.lo}, {args.hi}): {len(depths):,}; crop_half differs for {int(bad.sum()):,} "
          f"({bad.mean():.3e}); API minus training: +1 for {int((diff == 1).sum()):,}, -1 for "
          f"{int((diff == -1).sum()):,}, other {int((np.abs(diff) != 1).sum())}")
    if args.sample_csv:
        import pandas as pd
        d = pd.read_csv(args.sample_csv).depth_m.dropna().values
        # Weight each float32 depth by the sample's density there: the mismatch rate in each 0.5 m bin, averaged over
        # the sample's depths.
        edges = np.arange(args.lo, args.hi + 0.5, 0.5)
        idx = np.clip(np.digitize(depths, edges) - 1, 0, len(edges) - 2)
        rate = np.bincount(idx, weights=bad, minlength=len(edges) - 1) / np.maximum(np.bincount(idx, minlength=len(edges) - 1), 1)
        sidx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
        print(f"sample depths n={len(d)}, median {np.median(d):.2f} m; expected share of labels with a one-pixel "
              f"crop-half difference: {rate[sidx].mean():.3e}")


def figures(args):
    """The doc's two figures: crop-half disagreement by depth (from the exhaustive sweep), and the per-model change in
    p(incorrect) from a q95 WebP re-encode on the sampled crops (``webp-sample``'s per-label CSV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    os.makedirs(args.out, exist_ok=True)

    depths, bad = crop_half_sweep(1.0, 80.0)
    edges = np.arange(1.0, 80.5, 1.0)
    idx = np.clip(np.digitize(depths, edges) - 1, 0, len(edges) - 2)
    rate = np.bincount(idx, weights=bad, minlength=len(edges) - 1) / np.bincount(idx, minlength=len(edges) - 1)
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=110)
    ax.bar(edges[:-1] + 0.5, rate * 1e6, width=0.8, color=blue)
    ax.set_xlabel("depth at the view centre (m), 1 m bins", color=ink)
    ax.set_ylabel("per million depths", color=ink)
    ax.set_title("Crop half-size differs by one pixel (API float32 vs training float64)", color=ink, fontsize=10)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "crop_half_rounding.png"))
    plt.close(fig)

    if args.webp_csv:
        df = pd.read_csv(args.webp_csv)
        df["dp"] = (df.p_incorrect_gen2 - df.p_incorrect_gen1).abs()
        order = [t for t in LABEL_TYPES.values() if t in set(df.label_type)]
        fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=110)
        data = [np.log10(np.maximum(df[df.label_type == t].dp.values, 1e-8)) for t in order]
        parts = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                           flierprops={"marker": "o", "markersize": 3, "markerfacecolor": blue,
                                       "markeredgecolor": "none", "alpha": 0.5})
        for b in parts["boxes"]:
            b.set(facecolor="#d6e6f8", edgecolor=blue)
        for k in ("medians", "whiskers", "caps"):
            for line in parts[k]:
                line.set(color=blue)
        ax.set_yticks(range(1, len(order) + 1), order)
        ax.set_xticks(range(-8, 1), [f"1e{e}" for e in range(-8, 1)])
        ax.set_xlabel("|change in p(incorrect)| from one more q95 WebP generation (log scale)", color=ink)
        ax.set_title(f"Production validators on the {len(df):,} sampled crops", color=ink, fontsize=10)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(colors=muted)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "webp_gen2_dprob.png"))
        plt.close(fig)


def run(args):
    """All three stages: train worker(s), API worker, compare."""
    work = os.path.abspath(args.work)
    common = ["--pano", args.pano, "--depth-checkpoint", args.depth_checkpoint, "--api-dir", args.api_dir]
    if args.points:
        common += ["--points", args.points]
    me = os.path.abspath(__file__)
    subprocess.run([args.train_python, me, "train-worker", *common, "--out", os.path.join(work, "train"),
                    "--models", "--mps"], check=True)
    for i, py in enumerate(args.other_train_python or []):
        subprocess.run([py, me, "train-worker", *common, "--out", os.path.join(work, f"train_other{i + 1}"),
                        "--models", "--mps"], check=True)
    subprocess.run([args.api_python, me, "api-worker", "--api-dir", args.api_dir, "--depth-checkpoint",
                    args.depth_checkpoint, "--train-dir", os.path.join(work, "train"), "--out",
                    os.path.join(work, "api"), "--models"], check=True)
    others = [os.path.join(work, f"train_other{i + 1}") for i in range(len(args.other_train_python or []))]
    compare(argparse.Namespace(train=os.path.join(work, "train"), api=os.path.join(work, "api"), api_dir=args.api_dir,
                               out=args.out, train_other=others))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-worker", help="training side (training venv)")
    t.add_argument("--pano", required=True)
    t.add_argument("--depth-checkpoint", required=True)
    t.add_argument("--api-dir", required=True, help="sidewalk-ai-api checkout (model_map.py; read only)")
    t.add_argument("--points")
    t.add_argument("--out", required=True)
    t.add_argument("--models", action="store_true", help="also score the five production validators")
    t.add_argument("--mps", action="store_true", help="also render and run depth on mps (probes)")
    a = sub.add_parser("api-worker", help="serving side (API venv)")
    a.add_argument("--api-dir", required=True)
    a.add_argument("--depth-checkpoint", required=True)
    a.add_argument("--train-dir", required=True, help="the train worker's --out (points, store, stage-6 inputs)")
    a.add_argument("--out", required=True)
    a.add_argument("--models", action="store_true")
    a.add_argument("--pano-npy", help="cross-feed: replace the API's own read+resize with this (H, W, 3) BGR array, "
                                      "e.g. the train worker's pano_bgr.npy, so stages 2-4 run on identical input")
    c = sub.add_parser("compare", help="tables from the worker outputs")
    c.add_argument("--train", required=True)
    c.add_argument("--train-other", nargs="*", help="further train-worker outputs (other package pins)")
    c.add_argument("--api", required=True)
    c.add_argument("--api-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--append", nargs="*", help="text files (isolate, pixel-compare output) to append to summary.md")
    r = sub.add_parser("run", help="train worker, API worker and compare")
    r.add_argument("--train-python", default=sys.executable)
    r.add_argument("--other-train-python", nargs="*", help="interpreters with other training pins (stages 1-4 only)")
    r.add_argument("--api-python", required=True)
    r.add_argument("--api-dir", required=True)
    r.add_argument("--pano", required=True)
    r.add_argument("--points")
    r.add_argument("--depth-checkpoint", required=True)
    r.add_argument("--work", required=True, help="scratch directory for the workers' arrays (~1 GB)")
    r.add_argument("--out", required=True)
    w = sub.add_parser("webp-sample", help="second-generation WebP effect on the sampled crops")
    w.add_argument("--sample-csv", required=True)
    w.add_argument("--crops", required=True, help="the sample's crops/ directory (<LabelType>/<crop_path>)")
    w.add_argument("--api-dir", required=True)
    w.add_argument("--device", default="cpu")
    w.add_argument("--limit", type=int, help="rows per (type, split, class), for a quick run")
    w.add_argument("--out-csv", required=True, help="per-label output; keep it outside the repo")
    w.add_argument("--summary-csv")
    f = sub.add_parser("figures", help="the doc's figures")
    f.add_argument("--out", required=True)
    f.add_argument("--webp-csv", help="webp-sample's per-label CSV")
    x = sub.add_parser("isolate", help="stage-1 isolation from a cross-fed api-worker run")
    x.add_argument("--train", required=True)
    x.add_argument("--api", required=True, help="an api-worker --out made with --pano-npy <train>/pano_bgr.npy")
    x.add_argument("--out")
    pp = sub.add_parser("pixel-probe", help="decode/resize/WebP facts for this interpreter's builds")
    pp.add_argument("--pano", required=True)
    pp.add_argument("--reference-decode", help="another build's decode.npy, to resize identical input")
    pp.add_argument("--crops", nargs="*", help="PNG crops to WebP-encode")
    pp.add_argument("--crops-root", help="key the WebP hashes by path relative to this")
    pp.add_argument("--out", required=True)
    pc = sub.add_parser("pixel-compare", help="compare two pixel-probe outputs")
    pc.add_argument("a")
    pc.add_argument("b")
    pc.add_argument("--out")
    e = sub.add_parser("expected", help="expected /process outputs for a deploy smoke test")
    e.add_argument("--train", required=True)
    e.add_argument("--api", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--min-margin-mm", type=float, default=5.0,
                   help="cases whose centre depth is closer than this to a crop-half boundary leave the smoke set")
    h = sub.add_parser("crop-half", help="exhaustive float32 vs float64 crop-half disagreement")
    h.add_argument("--lo", type=float, default=1.0)
    h.add_argument("--hi", type=float, default=80.0)
    h.add_argument("--sample-csv", help="weight by this CSV's depth_m column")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    {"train-worker": train_worker, "api-worker": api_worker, "compare": compare, "run": run,
     "webp-sample": webp_sample, "crop-half": crop_half_report,
     "figures": figures, "isolate": isolate, "pixel-probe": pixel_probe, "pixel-compare": pixel_compare,
     "expected": expected_values}[args.cmd](args)


if __name__ == "__main__":
    main()
