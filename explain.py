"""Attention-rollout and Grad-CAM overlays explaining the validators' verdicts, plus the rater-agreement helper.

Issue #19: when the AI disagrees with a human label nothing says *why*. This script samples true positives, false
positives and false negatives from a held-out split, renders what the model "looked at" with two standard methods,
and writes contact sheets and a coding sheet for two human raters (rubric in ``docs/explanations-2026-10.md``).

**Categories.** The positive class is ``correct`` (the label is right; the API's Agree verdict). Against the folder
(crowd) class: TP = truth correct, predicted correct; FP = truth incorrect, predicted correct (the AI would Agree with
a bad label); FN = truth correct, predicted incorrect (the AI would Disagree with a good label); TN = both incorrect.
The verdict uses the API's tie rule (``calibration._isotonic_map.predict_verdicts``: ``incorrect`` only when
``p_incorrect > p_correct``). No 0.92 gate is applied: these are raw verdicts.

**Methods** (both on the 224 x 224 image the model actually sees: the saved ``AutoImageProcessor`` with
``use_fast=False`` resizes the short side to 256 and centre-crops 224, so the outer 6% of each side of the crop is
never seen; the overlays show that centre crop, not the full crop):

* ``attention_rollout`` (Abnar & Zuidema 2020, https://arxiv.org/abs/2005.00928): per layer, fuse the heads (mean by
  default), optionally zero the weakest ``discard_ratio`` of entries, add the identity for the residual connection,
  re-normalise rows, and multiply through the layers; the CLS row over the patch tokens is the map. The model is
  loaded with ``attn_implementation="eager"`` because SDPA returns no attention weights. It is class-agnostic.
* ``last_layer_cls_attention``: the last layer's CLS-to-patch attention per head (cheaper, also class-agnostic).
* ``grad_cam``: Grad-CAM (Selvaraju et al. 2017) adapted to a ViT the way pytorch-grad-cam does it: activations are
  the output of the last encoder block's ``norm1`` (the tokens entering its self-attention), channel weights are the
  mean gradient over the patch tokens, the map is the ReLU of the weighted sum, per patch. The target is the
  **logit margin** ``z[target] - z[other]`` (the log-odds of the verdict, which is what the softmax confidence is a
  function of), and ``target`` defaults to the predicted class. ``Dinov2ForImageClassification``'s head is a linear
  layer on ``[CLS ‖ mean(patch tokens)]`` of the final layernormed hidden state, so the gradient reaches the hooked
  tokens both through the last block's attention into CLS and through the mean-pool; the residual stream around
  ``norm1`` is not part of the hooked activation.

**Outputs** (``--out-dir``, one label type per run): ``predictions.csv`` (every crop in the split, with category),
``counts.json``, ``samples.csv`` (``overlay_id,label_uid,truth,pred,confidence,p_correct,category,overlay_path,
contact_sheet,tile``), one full-size PNG per sampled crop under ``overlays/`` (crop | rollout | Grad-CAM), and one
JPEG contact sheet per category (``<type>_<cat>.jpg``, downscaled tiles numbered as in ``samples.csv``). The crop is
centred on the label point (``data_generation.py``), so every crop panel carries a small cross at the centre.

**Private deployments.** The 2025 public datasets include crops from deployments that are private today. With
``--cities-json`` (the v3 API's city list) overlays are sampled only from public deployments and their label_uids use
the deployment's ``city_id``; only aggregate counts cover every crop. Always pass it for anything committed.

**Optional dependencies** (not in CI's install): torch, transformers (pinned 4.55.0 by requirements.txt), and
huggingface_hub for downloading the models. ``categorise``, ``sample_categories``, ``attention_rollout``, the rendering
helpers and ``agreement`` need only numpy/pandas/scikit-learn/matplotlib/pillow.

Usage (the exact commands behind docs/explanations-2026-10.md; ``$DATA`` holds ``<type>/test/<class>/*.webp``
downloaded from ``projectsidewalk/sidewalk-validator-ai-dataset-<type>`` at the revisions listed in the doc)::

    python explain.py run --label-type Crosswalk --crops-dir $DATA/crosswalk --out-dir $OUT/crosswalk \\
        --per-category 50 --seed 42 --device mps --check-device cpu --cities-json cities.json
    # ... the same for NoCurbRamp, Obstacle, SurfaceProblem
    python explain.py coding-sheet $OUT/{crosswalk,nocurbramp,obstacle,surfaceproblem}/samples.csv \\
        --out docs/explanations-2026-10/coding_sheet.csv
    # once two raters have each filled a copy of the coding sheet:
    python explain.py agreement coding_rater1.csv coding_rater2.csv

``--checkpoint`` defaults to the production model ``projectsidewalk/sidewalk-validator-ai-<type>`` at the revision
sidewalk-ai-api pins (``PROD_REVISIONS``). CurbRamp is accepted but its public dataset is a copy of the Crosswalk one
(issue #7), so explaining it on those crops says nothing about curb ramps.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from calibration._isotonic_map import predict_verdicts

CLASSES = ("correct", "incorrect")
POSITIVE = "correct"
CATEGORIES = ("TP", "FP", "FN", "TN")
DEFAULT_CATEGORIES = ("TP", "FP", "FN")
LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")

PROD_REVISIONS = {
    "CurbRamp": "969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a",
    "NoCurbRamp": "9f3f3bb2f40864a468fb65e43644ee5ba54e3e19",
    "Obstacle": "8f0fe8fc48748fac35098e74c1848d015126deef",
    "SurfaceProblem": "0e1439daa5667c8a81f63e75f835bd51221df4c2",
    "Crosswalk": "b7936e5bd17a4b96db301f954e5d3a5fe9eaf361",
}
"""Model revisions sidewalk-ai-api's model_map.py pins (docs/calibration-2026-10.md, "Scope")."""

# Coding-sheet vocabulary (the rubric in docs/explanations-2026-10.md explains each value).
COVERS = ("yes", "partly", "no", "unclear")
OTHER_HIGHLIGHT = ("none", "sidewalk_surface", "road_surface", "curb_edge", "crosswalk_paint", "vehicle",
                   "pole_or_sign", "vegetation", "building_or_wall", "person", "sky", "image_corner_or_border",
                   "scattered_background", "other")
INFORMATIVE = ("yes", "no")
CODED_COLUMNS = ("rollout_covers", "gradcam_covers", "other_highlight", "informative")
ORDINAL_COLUMNS = {"rollout_covers": ("no", "partly", "yes"), "gradcam_covers": ("no", "partly", "yes")}
SHEET_ID_COLUMNS = ["label_type", "overlay_id", "label_uid", "category", "truth", "pred", "confidence",
                    "contact_sheet", "tile"]
SHEET_RATER_COLUMNS = ["rater", *CODED_COLUMNS, "notes"]


def default_checkpoint(label_type):
    """The production model repo for ``label_type`` (``projectsidewalk/sidewalk-validator-ai-crosswalk``)."""
    return f"projectsidewalk/sidewalk-validator-ai-{label_type.lower()}"


# ---------------------------------------------------------------------------------------------------------------------
# Input and categories -- pure, unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def load_crops_dir(crops_dir, split="test"):
    """Manifest of ``<crops_dir>/<split>/<class>/<city>_<label_id>.<ext>`` (the Hugging Face dataset layout).

    ``label_uid`` is ``<city>:<label_id>``: label ids are per-city, so the city is always part of the key.

    >>> load_crops_dir("/nonexistent")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    FileNotFoundError: no images under /nonexistent/test/{correct,incorrect}/
    """
    rows = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(crops_dir, split, cls, "*"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            city, _, label_id = stem.rpartition("_")
            rows.append({"label_uid": f"{city}:{label_id}", "city": city, "label_id": label_id,
                         "truth": cls, "crop_path": path})
    if not rows:
        raise FileNotFoundError(f"no images under {crops_dir}/{split}/{{correct,incorrect}}/")
    df = pd.DataFrame(rows)
    if not df.label_uid.is_unique:
        raise ValueError(f"{crops_dir}/{split}: label_uid is not unique")
    return df


def load_split_csv(path, split="test"):
    """Manifest from a ``split_data.py`` ``split.csv`` (rows of ``split``); ``crop_path`` relative to the CSV's dir."""
    df = pd.read_csv(path)
    df = df[df.split == split].copy()
    if not df.label_uid.is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    base = os.path.dirname(os.path.abspath(path))
    df["crop_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df.crop_path]
    return df.rename(columns={"label_class": "truth"})[["label_uid", "city", "label_id", "truth", "crop_path"]]


DATASET_CITY_ALIASES = {"sea": "seattle-wa"}
"""Dataset file prefixes that are not a prefix of their deployment's ``city_id`` (the 2025 datasets wrote ``sea_``)."""


def city_ids(prefixes, cities_json):
    """``{dataset prefix: city_id}`` for the prefixes that are **public** deployments in ``cities_json``.

    The published crops are named ``<prefix>_<label_id>.webp`` with a shortened, underscored city (``chicago``,
    ``new_taipei``, ``sea``). A prefix matches the public ``city_id`` that equals it with ``_`` -> ``-``, or starts with
    that plus ``-`` (``chicago`` -> ``chicago-il``), or ``DATASET_CITY_ALIASES``. Anything else, including every
    private deployment, is left out: callers drop those rows, so an unrecognised city fails closed.

    ``cities_json`` is the v3 API's ``/v3/api/cities`` response (``{"cities": [{"city_id", "visibility"}, ...]}``).

    >>> cj = {"cities": [{"city_id": "chicago-il", "visibility": "public"}, {"city_id": "taipei", "visibility": "private"},
    ...                  {"city_id": "seattle-wa", "visibility": "public"}]}
    >>> city_ids(["chicago", "taipei", "sea", "nowhere"], cj)
    {'chicago': 'chicago-il', 'sea': 'seattle-wa'}
    """
    if isinstance(cities_json, str):
        with open(cities_json) as f:
            cities_json = json.load(f)
    cities = cities_json.get("cities", cities_json) if isinstance(cities_json, dict) else cities_json
    public = sorted(c["city_id"] for c in cities if c.get("visibility") == "public")
    out = {}
    for prefix in prefixes:
        alias = DATASET_CITY_ALIASES.get(prefix)
        dashed = prefix.replace("_", "-")
        hits = [c for c in public if c == alias or c == dashed or c.startswith(dashed + "-")]
        if len(hits) == 1:
            out[prefix] = hits[0]
    return out


def categorise(truth, pred, positive=POSITIVE):
    """TP/FP/FN/TN of each (truth, pred) pair with ``positive`` as the positive class.

    >>> categorise(["correct", "incorrect", "correct", "incorrect"], ["correct", "correct", "incorrect", "incorrect"])
    ['TP', 'FP', 'FN', 'TN']
    """
    out = []
    for t, p in zip(truth, pred):
        if t not in CLASSES or p not in CLASSES:
            raise ValueError(f"unexpected class pair ({t!r}, {p!r})")
        if p == positive:
            out.append("TP" if t == positive else "FP")
        else:
            out.append("FN" if t == positive else "TN")
    return out


def add_verdicts(df):
    """Add ``pred``, ``confidence`` and ``category`` to a frame with ``truth, p_correct, p_incorrect``."""
    df = df.copy()
    pred, conf = predict_verdicts(df.p_correct.values, df.p_incorrect.values)
    df["pred"] = pred
    df["confidence"] = conf
    df["category"] = categorise(df.truth, df.pred)
    return df


def sample_categories(df, per_category, seed=42, categories=DEFAULT_CATEGORIES, label_type=""):
    """Up to ``per_category`` rows of each category, sampled with ``seed``; adds ``overlay_id`` and ``tile``.

    Rows are first sorted by ``label_uid`` so the sample depends only on the data and the seed, not on file order.
    Within a category the sample is ordered by descending confidence (the most confident mistakes first), and
    ``overlay_id`` is ``<type>-<cat>-<NN>`` with ``tile`` = NN, the number printed on the contact-sheet tile.

    Example::

        s = sample_categories(predictions, 50, seed=42, label_type="Crosswalk")
        s.groupby("category").size()   # at most 50 each; fewer when the category is smaller
    """
    df = df.sort_values("label_uid").reset_index(drop=True)
    prefix = label_type.lower() or "x"
    parts = []
    for i, cat in enumerate(categories):
        rows = df[df.category == cat]
        if len(rows) > per_category:
            rows = rows.sample(n=per_category, random_state=seed + i)
        rows = rows.sort_values(["confidence", "label_uid"], ascending=[False, True]).copy()
        rows["tile"] = np.arange(1, len(rows) + 1)
        rows["overlay_id"] = [f"{prefix}-{cat}-{n:02d}" for n in rows.tile]
        parts.append(rows)
    if not parts:
        return df.iloc[:0].assign(tile=[], overlay_id=[])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
# Explanation maps -- rollout is pure numpy; Grad-CAM needs torch
# ---------------------------------------------------------------------------------------------------------------------

def attention_rollout(attentions, head_fusion="mean", discard_ratio=0.0, num_prefix_tokens=1):
    """Attention rollout: the CLS token's rolled-out attention to each patch token.

    Args:
        attentions: one array per layer, each ``(heads, T, T)`` (or ``(1, heads, T, T)``), rows summing to 1, token
            0 being CLS. Tensors are accepted and converted with ``np.asarray``.
        head_fusion: ``mean``, ``max`` or ``min`` over heads.
        discard_ratio: share of the smallest fused entries per layer to zero before adding the identity (Abnar &
            Zuidema use 0; pytorch-grad-cam's example 0.9). The CLS-to-CLS entry is never discarded.
        num_prefix_tokens: CLS (+ register) tokens before the patches; DINOv2 without registers has 1.

    Returns:
        ``(T - num_prefix_tokens,)`` array: rollout from CLS to each patch, in token order (reshape with
        ``to_grid``). Not rescaled; the full CLS row of the rollout matrix sums to 1.

    Example: with identity attention in every layer nothing mixes, so CLS attends only to itself::

        >>> eye = np.eye(5)[None]                       # 1 head, CLS + 4 patches
        >>> attention_rollout([eye, eye]).tolist()
        [0.0, 0.0, 0.0, 0.0]
    """
    fuse = {"mean": np.mean, "max": np.max, "min": np.min}[head_fusion]
    result = None
    for a in attentions:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 4:
            if a.shape[0] != 1:
                raise ValueError("attention_rollout takes one image at a time")
            a = a[0]
        fused = fuse(a, axis=0)
        t = fused.shape[-1]
        if discard_ratio > 0:
            flat = fused.reshape(-1).copy()
            k = int(flat.size * discard_ratio)
            order = np.argsort(flat, kind="stable")[:k]
            order = order[order != 0]                      # keep CLS -> CLS
            flat[order] = 0
            fused = flat.reshape(t, t)
        a = 0.5 * fused + 0.5 * np.eye(t)                  # residual connection
        a = a / a.sum(axis=-1, keepdims=True)
        result = a if result is None else a @ result
    if result is None:
        raise ValueError("no attention layers")
    return result[0, num_prefix_tokens:]


def last_layer_cls_attention(attentions, num_prefix_tokens=1):
    """``(heads, T - num_prefix_tokens)`` CLS-to-patch attention of the last layer; ``.mean(0)`` for one map."""
    a = np.asarray(attentions[-1], dtype=np.float64)
    if a.ndim == 4:
        a = a[0]
    return a[:, 0, num_prefix_tokens:]


def to_grid(values):
    """Reshape a per-patch vector of length n*n to an ``(n, n)`` grid (row-major, as DINOv2 orders patches)."""
    values = np.asarray(values)
    n = int(round(np.sqrt(values.size)))
    if n * n != values.size:
        raise ValueError(f"{values.size} patches is not a square grid")
    return values.reshape(n, n)


def centre_mass(grid, frac=0.5):
    """Share of a non-negative map's total that falls in the central ``frac`` x ``frac`` box of the grid.

    The label point is at the centre of every crop, so this is a crude, automatic stand-in for "does the map sit on
    the labelled feature": a map spread uniformly scores ``frac**2`` (0.25 by default), one on the centre scores 1.
    Features are not all centred or the same size, which is why the raters' coding, not this number, answers #19.

    >>> g = np.zeros((4, 4)); g[1:3, 1:3] = 1
    >>> centre_mass(g)
    1.0
    >>> centre_mass(np.ones((4, 4)))
    0.25
    """
    g = np.clip(np.asarray(grid, dtype=np.float64), 0, None)
    n = g.shape[0]
    k = max(1, int(round(n * frac)))
    lo = (n - k) // 2
    total = g.sum()
    return float(g[lo:lo + k, lo:lo + k].sum() / total) if total > 0 else float("nan")


def top_mass(values, k=3):
    """Share of a non-negative map's total held by its ``k`` largest patches (how peaked it is).

    >>> top_mass([1, 1, 1, 1], k=1)
    0.25
    """
    v = np.sort(np.clip(np.asarray(values, dtype=np.float64).ravel(), 0, None))[::-1]
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else float("nan")


def explain_one(model, pixel_values, target_class=None, num_prefix_tokens=1):
    """One forward + backward pass: probabilities, attention rollout, last-layer CLS attention and Grad-CAM.

    ``model`` is a ``Dinov2ForImageClassification`` loaded with ``attn_implementation="eager"``; ``pixel_values`` is
    ``(1, 3, H, W)`` on the model's device. ``target_class`` (an index) defaults to the predicted class (argmax, the
    first index on a tie, as the API does). The Grad-CAM score is the logit margin ``z[target] - z[other]``.

    Returns:
        dict with ``probs`` (numpy, per class index), ``target``, ``rollout``, ``cls_attention`` (heads x patches),
        ``gradcam`` (patches), each per patch in token order.
    """
    import torch

    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["act"] = output

    block = model.dinov2.encoder.layer[-1]
    handle = block.norm1.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(pixel_values=pixel_values, output_attentions=True)
            if out.attentions is None or out.attentions[0] is None:
                raise RuntimeError('no attention weights: load the model with attn_implementation="eager"')
            logits = out.logits[0]
            target = int(logits.argmax()) if target_class is None else int(target_class)
            if logits.numel() == 2:
                score = logits[target] - logits[1 - target]
            else:
                score = logits[target] - torch.logsumexp(torch.cat([logits[:target], logits[target + 1:]]), 0)
            score.backward()
        act = captured["act"][0, num_prefix_tokens:].detach()
        grad = captured["act"].grad[0, num_prefix_tokens:].detach()
        weights = grad.mean(dim=0)
        cam = torch.relu(act @ weights).float().cpu().numpy()
        attentions = [a.detach().float().cpu().numpy() for a in out.attentions]
        probs = logits.detach().softmax(-1).float().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return {"probs": probs, "target": target,
            "rollout": attention_rollout(attentions, num_prefix_tokens=num_prefix_tokens),
            "cls_attention": last_layer_cls_attention(attentions, num_prefix_tokens),
            "gradcam": cam}


def grad_cam(model, pixel_values, target_class=None):
    """Grad-CAM per patch (see ``explain_one``); returns ``(cam, target_index)``."""
    r = explain_one(model, pixel_values, target_class)
    return r["gradcam"], r["target"]


# ---------------------------------------------------------------------------------------------------------------------
# Model I/O -- needs torch + transformers; not covered by CI
# ---------------------------------------------------------------------------------------------------------------------

def load_model(checkpoint, revision=None, device="cpu"):
    """``(image_processor, model)``: the saved slow processor and the classifier with eager attention, in eval mode."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(checkpoint, revision=revision, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, revision=revision,
                                                            attn_implementation="eager").to(device).eval()
    if set(model.config.id2label.values()) != set(CLASSES):
        raise ValueError(f"unexpected id2label {model.config.id2label}")
    return processor, model


def class_index(model):
    """``{class name: logit index}`` from the model's ``id2label``."""
    return {v: int(k) for k, v in model.config.id2label.items()}


def predict(model, processor, paths, device="cpu", batch_size=16):
    """``(n, 2)`` softmax probabilities in ``CLASSES`` order for the images at ``paths``."""
    import torch
    from tqdm import tqdm

    idx = class_index(model)
    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"predict ({device})"):
        images = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = model(**inputs).logits.softmax(-1).float().cpu().numpy()
        out.append(probs[:, [idx[c] for c in CLASSES]])
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ---------------------------------------------------------------------------------------------------------------------
# Rendering -- pure (numpy / matplotlib / PIL), unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def denormalise(pixel_values, mean, std):
    """``(H, W, 3)`` uint8 image from one ``(3, H, W)`` normalised tensor/array (undoes the processor)."""
    x = np.asarray(pixel_values, dtype=np.float64)
    x = x * np.asarray(std)[:, None, None] + np.asarray(mean)[:, None, None]
    return (np.clip(x.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)


def heat_overlay(image, grid, alpha=0.5, cmap="inferno"):
    """Blend a patch grid (upsampled bilinearly, min-max scaled) over ``image`` (H, W, 3 uint8)."""
    h, w = image.shape[:2]
    g = np.asarray(grid, dtype=np.float64)
    span = g.max() - g.min()
    g = (g - g.min()) / span if span > 0 else np.zeros_like(g)
    up = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    colour = plt.get_cmap(cmap)(up)[..., :3] * 255
    return (alpha * colour + (1 - alpha) * image).round().astype(np.uint8)


def mark_centre(image, size=6, colour=(0, 255, 255)):
    """Copy of ``image`` with a small cross at the centre (where the label point is in every crop)."""
    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    cx, cy = im.width // 2, im.height // 2
    d.line([(cx - size, cy), (cx + size, cy)], fill=colour, width=1)
    d.line([(cx, cy - size), (cx, cy + size)], fill=colour, width=1)
    return np.asarray(im)


def strip(image, maps):
    """Side-by-side ``crop | overlay(map) ...`` as one uint8 array; ``maps`` are patch grids."""
    panels = [mark_centre(image)] + [heat_overlay(image, m) for m in maps]
    return np.concatenate(panels, axis=1)


def write_overlay(path, image, maps, title, names=("rollout", "Grad-CAM")):
    """Full-size PNG: the model's view with a centre cross, then one overlay per map, with a title line."""
    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(3.2 * (1 + len(maps)), 3.6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(mark_centre(image))
    axes[0].set_title("model input (224 centre crop)", fontsize=8)
    for ax, m, name in zip(axes[1:], maps, names):
        ax.imshow(heat_overlay(image, m))
        ax.set_title(name, fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 10.1 has no scalable default font
        return ImageFont.load_default()


def contact_sheet(strips, captions, path, ncols=4, tile_height=112, header=None, quality=82):
    """Grid of downscaled strips with a caption above each; JPEG (photographs compress ~10x better than PNG).

    ``strips`` are uint8 arrays of equal height (from ``strip``); each is resized to ``tile_height`` px high.
    """
    font, head_font = _font(11), _font(14)
    cap_h, pad = 16, 6
    tiles = []
    for s in strips:
        im = Image.fromarray(s)
        w = round(im.width * tile_height / im.height)
        tiles.append(im.resize((w, tile_height), Image.LANCZOS))
    tile_w = max((t.width for t in tiles), default=3 * tile_height)
    nrows = max(1, -(-len(tiles) // ncols))
    head_h = 24 if header else 0
    sheet = Image.new("RGB", (ncols * (tile_w + pad) + pad, head_h + nrows * (tile_height + cap_h + pad) + pad),
                      "white")
    d = ImageDraw.Draw(sheet)
    if header:
        d.text((pad, 4), header, fill="black", font=head_font)
    for i, (t, c) in enumerate(zip(tiles, captions)):
        r, k = divmod(i, ncols)
        x, y = pad + k * (tile_w + pad), head_h + pad + r * (tile_height + cap_h + pad)
        d.text((x, y), c, fill="black", font=font)
        sheet.paste(t, (x, y + cap_h))
    if not tiles:
        d.text((pad, head_h + pad), "(no examples in this category)", fill="black", font=font)
    sheet.save(path, quality=quality, optimize=True)
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Coding sheet and agreement -- pure, unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def coding_sheet(samples):
    """Blank coding sheet (one row per sampled overlay) from concatenated ``samples.csv`` frames.

    Raters each fill a copy; ``rater`` is a role (``rater1``), never a name or account id.
    """
    sheet = samples[SHEET_ID_COLUMNS].copy()
    for c in SHEET_RATER_COLUMNS:
        sheet[c] = ""
    return sheet


def validate_codes(sheet):
    """Raise if a filled sheet uses a value outside the rubric's vocabulary (blank cells are allowed)."""
    allowed = {"rollout_covers": COVERS, "gradcam_covers": COVERS, "other_highlight": OTHER_HIGHLIGHT,
               "informative": INFORMATIVE}
    for col, values in allowed.items():
        bad = set(sheet[col].dropna().astype(str).str.strip()) - set(values) - {""}
        if bad:
            raise ValueError(f"{col}: values {sorted(bad)} are not in the rubric ({', '.join(values)})")


def agreement(sheet1, sheet2, columns=CODED_COLUMNS, by="label_type"):
    """Cohen's kappa between two raters' sheets, per coded column, overall and per ``by`` group.

    Rows are joined on ``overlay_id`` (one-to-one); a row counts for a column only when both raters coded it and
    neither said ``unclear``. The ordinal ``*_covers`` columns also get a linear-weighted kappa over no < partly < yes.

    Returns:
        DataFrame ``column, group, n, percent_agreement, kappa, weighted_kappa``.

    Example (hand-computed: 3 of 4 agree, so p_o = 0.75; rater 1 says yes 2/4, rater 2 3/4, so
    p_e = 0.5 * 0.75 + 0.5 * 0.25 = 0.5; kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5)::

        >>> a = pd.DataFrame({"overlay_id": list("abcd"), "label_type": "X", "informative": ["yes", "yes", "no", "no"]})
        >>> b = a.assign(informative=["yes", "yes", "no", "yes"])
        >>> float(agreement(a, b, ["informative"]).kappa.iloc[0])
        0.5
    """
    from sklearn.metrics import cohen_kappa_score

    merged = sheet1.merge(sheet2, on="overlay_id", suffixes=("_1", "_2"), validate="one_to_one")
    group_col = f"{by}_1" if f"{by}_1" in merged.columns else by
    rows = []
    for col in columns:
        a = merged[f"{col}_1"].fillna("").astype(str).str.strip()
        b = merged[f"{col}_2"].fillna("").astype(str).str.strip()
        ok = (a != "") & (b != "") & (a != "unclear") & (b != "unclear")
        groups = [("all", ok)]
        if group_col in merged.columns:
            groups += [(g, ok & (merged[group_col] == g)) for g in sorted(merged[group_col].dropna().unique())]
        for name, mask in groups:
            x, y = a[mask].tolist(), b[mask].tolist()
            n = len(x)
            row = {"column": col, "group": name, "n": n,
                   "percent_agreement": float(np.mean([i == j for i, j in zip(x, y)])) if n else np.nan,
                   "kappa": np.nan, "weighted_kappa": np.nan}
            if n and len(set(x) | set(y)) > 1:
                row["kappa"] = float(cohen_kappa_score(x, y))
                if col in ORDINAL_COLUMNS:
                    order = {v: i for i, v in enumerate(ORDINAL_COLUMNS[col])}
                    row["weighted_kappa"] = float(cohen_kappa_score([order[v] for v in x], [order[v] for v in y],
                                                                    labels=list(order.values()), weights="linear"))
            elif n:
                row["kappa"] = 1.0 if row["percent_agreement"] == 1.0 else np.nan   # one value only: undefined
            rows.append(row)
    return pd.DataFrame(rows, columns=["column", "group", "n", "percent_agreement", "kappa", "weighted_kappa"])


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def run(args):
    import torch

    label_type = args.label_type
    checkpoint = args.checkpoint or default_checkpoint(label_type)
    revision = args.revision or (PROD_REVISIONS[label_type] if not args.checkpoint else None)
    os.makedirs(os.path.join(args.out_dir, "overlays"), exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    manifest = load_split_csv(args.split_csv, args.split) if args.split_csv else load_crops_dir(args.crops_dir,
                                                                                               args.split)
    print(f"{label_type}: {len(manifest)} crops in split {args.split}; model {checkpoint} @ {revision}; "
          f"device {args.device}")
    processor, model = load_model(checkpoint, revision, args.device)
    probs = predict(model, processor, manifest.crop_path.tolist(), args.device, args.batch_size)
    preds = add_verdicts(manifest.assign(p_correct=probs[:, 0], p_incorrect=probs[:, 1]))
    preds.drop(columns=["crop_path"]).to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    counts = {c: int((preds.category == c).sum()) for c in CATEGORIES}
    pool = preds
    if args.cities_json:
        mapping = city_ids(sorted(preds.city.unique()), args.cities_json)
        pool = preds[preds.city.isin(mapping)].copy()
        pool["label_uid"] = [f"{mapping[c]}:{i}" for c, i in zip(pool.city, pool.label_id)]
        print(f"sampling from public deployments only: {len(pool)} of {len(preds)} crops "
              f"({len(set(preds.city) - set(mapping))} dataset cities left out)")
    public_counts = {c: int((pool.category == c).sum()) for c in CATEGORIES}
    info = {"label_type": label_type, "checkpoint": checkpoint, "revision": revision, "device": args.device,
            "split": args.split, "n": int(len(preds)), "categories": counts,
            "public_only": bool(args.cities_json), "n_sampling_pool": int(len(pool)),
            "categories_sampling_pool": public_counts,
            "attn_implementation": model.config._attn_implementation, "torch": torch.__version__,
            "seed": args.seed, "per_category": args.per_category}
    print("categories:", counts)

    if args.check_device:
        _, ref_model = load_model(checkpoint, revision, args.check_device)
        ref = predict(ref_model, processor, manifest.crop_path.tolist(), args.check_device, args.batch_size)
        ref_cat = add_verdicts(manifest.assign(p_correct=ref[:, 0], p_incorrect=ref[:, 1])).category.values
        info["device_check"] = {"against": args.check_device,
                                "max_abs_diff_p_correct": float(np.abs(ref[:, 0] - probs[:, 0]).max()),
                                "category_flips": int((ref_cat != preds.category.values).sum())}
        print("device check:", info["device_check"])
        del ref_model

    categories = tuple(c.strip() for c in args.categories.split(","))
    samples = sample_categories(pool, args.per_category, args.seed or 0, categories, label_type)
    mean, std = processor.image_mean, processor.image_std
    method_names = {"rollout": ("rollout",), "gradcam": ("Grad-CAM",), "both": ("rollout", "Grad-CAM")}[args.method]
    idx = class_index(model)
    strips, max_dp = {c: [] for c in categories}, 0.0
    overlay_paths = []
    stats = {f"{m}_{s}": [] for m in ("rollout", "gradcam") for s in ("centre_mass", "top3_mass")}
    for row in samples.itertuples():
        inputs = processor(Image.open(row.crop_path).convert("RGB"), return_tensors="pt").to(args.device)
        target = None if args.target == "pred" else idx[args.target]
        r = explain_one(model, inputs["pixel_values"], target)
        max_dp = max(max_dp, abs(float(r["probs"][idx["correct"]]) - row.p_correct))
        image = denormalise(inputs["pixel_values"][0].cpu().numpy(), mean, std)
        maps = {"rollout": to_grid(r["rollout"]), "Grad-CAM": to_grid(r["gradcam"])}
        for key, name in (("rollout", "rollout"), ("gradcam", "Grad-CAM")):
            stats[f"{key}_centre_mass"].append(centre_mass(maps[name]))
            stats[f"{key}_top3_mass"].append(top_mass(maps[name]))
        chosen = [maps[m] for m in method_names]
        rel = os.path.join("overlays", f"{row.overlay_id}.png")
        title = (f"{row.overlay_id}  {row.label_uid}  truth {row.truth}  pred {row.pred} "
                 f"({row.confidence:.4f})  {row.category}")
        write_overlay(os.path.join(args.out_dir, rel), image, chosen, title, method_names)
        overlay_paths.append(rel)
        strips[row.category].append((strip(image, chosen),
                                     f"{row.tile:02d} {row.label_uid} {row.pred[:3]} {row.confidence:.3f}"))
    samples["overlay_path"] = overlay_paths
    for key, values in stats.items():
        samples[key] = values
    samples["contact_sheet"] = [f"{label_type.lower()}_{c}.jpg" for c in samples.category]
    info["explain_pass_max_abs_diff_p_correct"] = max_dp
    for cat in categories:
        items = strips[cat]
        header = (f"{label_type} {cat} (n={len(items)} of {public_counts[cat]}): model input with label-point cross | "
                  + " | ".join(method_names) + ". Caption: tile, label_uid, verdict, confidence")
        contact_sheet([s for s, _ in items], [c for _, c in items],
                      os.path.join(args.out_dir, f"{label_type.lower()}_{cat}.jpg"), ncols=args.ncols, header=header)
    cols = ["overlay_id", "label_uid", "truth", "pred", "confidence", "p_correct", "category", "overlay_path",
            "contact_sheet", "tile", *stats]
    samples.assign(label_type=label_type)[["label_type", *cols]].to_csv(os.path.join(args.out_dir, "samples.csv"),
                                                                         index=False, float_format="%.6f")
    with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="predict a split, sample TP/FP/FN, render overlays and contact sheets")
    p.add_argument("--label-type", required=True, choices=LABEL_TYPES)
    p.add_argument("--checkpoint", default=None,
                   help="local checkpoint or HF model id (default: the production repo for --label-type)")
    p.add_argument("--revision", default=None,
                   help="HF revision for --checkpoint (default with no --checkpoint: the pinned production revision)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--crops-dir", help="directory in the HF dataset layout: <dir>/<split>/<class>/<city>_<id>.webp")
    src.add_argument("--split-csv", help="split_data.py split.csv (crop_path relative to its directory)")
    p.add_argument("--split", default="test", help="which split to explain (default test)")
    p.add_argument("--per-category", type=int, default=50, help="maximum overlays per category (default 50)")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="comma-separated subset of TP,FP,FN,TN (positive class = correct; default TP,FP,FN)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    p.add_argument("--device", default="cpu", help="torch device for all passes: cpu or mps (default cpu)")
    p.add_argument("--check-device", default=None,
                   help="also predict on this device and report max |dp| and category flips against --device")
    p.add_argument("--batch-size", type=int, default=16, help="batch size of the prediction pass (default 16)")
    p.add_argument("--method", choices=("rollout", "gradcam", "both"), default="both",
                   help="which maps to render next to the crop (default both)")
    p.add_argument("--target", choices=("pred", *CLASSES), default="pred",
                   help="Grad-CAM target class: the predicted one (default) or a fixed class")
    p.add_argument("--ncols", type=int, default=4, help="tiles per row in the contact sheets (default 4)")
    p.add_argument("--cities-json", default=None,
                   help="v3 API /v3/api/cities response; when given, overlays are sampled only from public "
                        "deployments and label_uids use their city_id (seattle-wa:123). Counts in counts.json still "
                        "cover every crop. Use it for anything that is committed or shared")
    p.add_argument("--out-dir", required=True, help="output directory for this label type")

    c = sub.add_parser("coding-sheet", help="blank coding sheet from one or more samples.csv")
    c.add_argument("samples", nargs="+", help="samples.csv files written by `run`")
    c.add_argument("--out", required=True, help="CSV to write")

    a = sub.add_parser("agreement", help="Cohen's kappa between two raters' filled coding sheets")
    a.add_argument("sheet1")
    a.add_argument("sheet2")
    a.add_argument("--out", default=None, help="optional CSV for the agreement table")

    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "coding-sheet":
        sheet = coding_sheet(pd.concat([pd.read_csv(s) for s in args.samples], ignore_index=True))
        sheet.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(sheet)} rows)")
    else:
        s1, s2 = pd.read_csv(args.sheet1, dtype=str), pd.read_csv(args.sheet2, dtype=str)
        validate_codes(s1)
        validate_codes(s2)
        table = agreement(s1, s2)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if args.out:
            table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
