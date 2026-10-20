"""Saliency overlays for the validators' verdicts, a blinded packet for two human raters, and their agreement.

Issue #19: when the AI disagrees with a human label nothing says *why*. This script samples true positives, false
positives, false negatives and (as a control) true negatives from a held-out split, computes four saliency maps per
crop, and builds (a) labelled contact sheets for readers and (b) a **blinded** packet for two raters, whose coding is
unblinded and scored by ``agreement`` (rubric in ``docs/explanations-2026-10.md``).

**Categories.** The positive class is ``correct`` (the label is right; the API's Agree verdict). Against the folder
(crowd) class: TP = truth correct, predicted correct; FP = truth incorrect, predicted correct; FN = truth correct,
predicted incorrect; TN = both incorrect. The verdict uses the API's tie rule
(``calibration._isotonic_map.predict_verdicts``); no 0.92 gate is applied.

**What the model sees.** The saved ``AutoImageProcessor`` (``use_fast=False``) resizes the short side to 256 and
centre-crops 224. A square 640 px crop is centred on the label point (``data_generation.py``), so the label point is
the centre of the 224 view and the outer 1/16 of each side is cut. About 9% of the 2025 crops are **not square**
(640 x h, h < 640: the crop was clipped at the view border), so their label point is off-centre by an unknown amount
and up to ~22% per side is cut horizontally. Those rows carry ``label_point_uncertain = True``; their panels have no
centre cross and they are left out of every centre/border statistic.

**Maps** (all per 14 px patch on the 16 x 16 grid, row-major, CLS dropped):

* ``rollout`` (Abnar & Zuidema 2020): head-mean attention per layer, identity added for the residual, rows
  re-normalised, multiplied through the layers; the CLS row. Class-agnostic. Needs ``attn_implementation="eager"``.
* ``cls_attention``: the last layer's CLS-to-patch attention, mean over heads. Class-agnostic.
* ``gradcam_<class>``: Grad-CAM as pytorch-grad-cam adapts it to a ViT. Activations are the output of the last
  block's ``norm1``; one channel-weight vector (the gradient averaged over patches) for the whole image; ReLU of the
  weighted sum. Target = the logit margin ``z[class] - z[other]``. Computed for **both** classes.
* ``gradxact_<class>``: per-token gradient x activation at the last block's **output** (the residual stream the
  final layernorm reads), summed over channels, ReLU. Same target. Unlike Grad-CAM it has per-token weights.
* ``centre_blob``: a fixed Gaussian at the grid centre (sigma 2.5 patches). A control, not an explanation: any map
  that "covers the feature" no more often than this blob is not shown to localise.

``Dinov2ForImageClassification``'s head reads ``[CLS || mean(patch tokens)]`` of the final layernormed hidden state,
so the block-output patch tokens reach the logits only through the mean-pool (grad x act attributes that path), and
CLS reaches them through attention (which Grad-CAM at ``norm1`` sees).

**Outputs of ``run``** (``--out-dir``, one label type): ``predictions.csv`` (every crop, private deployments
included when present: keep it out of the repo), ``counts.json``, ``samples.csv`` (one row per sampled crop with the
map statistics), ``maps/<overlay_id>.npz`` (the 224 view and every map; input to ``packet``) and one JPEG contact
sheet per category (``<type>_<cat>.jpg``: crop | rollout | CLS attention | Grad-CAM | grad x act, the last two for the
predicted class).

**Optional dependencies** (not in CI's install): torch, transformers (4.55.0 per requirements.txt), and
huggingface_hub to download the models. Everything except ``run`` needs only numpy/pandas/scikit-learn/matplotlib/
pillow.

Usage (``$DATA/<type>/test/<class>/*.webp`` from ``projectsidewalk/sidewalk-validator-ai-dataset-<type>``)::

    python explain.py run --label-type Crosswalk --crops-dir $DATA/crosswalk --out-dir $OUT/crosswalk \\
        --cities-json cities.json --per-category 50 --seed 42 --device mps --check-device cpu
    # ... the same for NoCurbRamp, Obstacle, SurfaceProblem
    python explain.py summary $OUT/*/samples.csv
    python explain.py packet $OUT/crosswalk $OUT/nocurbramp $OUT/obstacle $OUT/surfaceproblem \\
        --out-dir $PACKET --key $PACKET_KEY_DIR/packet_key.csv --seed $SECRET_SEED
    # once two raters have each filled a copy of $PACKET/coding_sheet.csv:
    python explain.py agreement coding_rater1.csv coding_rater2.csv --key $PACKET_KEY_DIR/packet_key.csv \\
        --key-sha256 docs/explanations-2026-10/packet_key.sha256

``--cities-json`` (the v3 API's ``/v3/api/cities``) is required: overlays are sampled only from public deployments,
and ``label_uid`` is ``<city_id>:<label_id>``. ``--include-private`` drops the filter for local debugging only; its
rows get ``label_uid = dataset:<file stem>`` so they can never be mistaken for a deployment-qualified id.
``--checkpoint`` defaults to the production model at the revision sidewalk-ai-api pins (``PROD_REVISIONS``).
CurbRamp is accepted, but its public dataset is a copy of the Crosswalk one (issue #7).
"""
import argparse
import glob
import json
import os
import time

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
DEFAULT_CATEGORIES = CATEGORIES
AGREE_CATEGORIES = ("TP", "FP")
LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")

PROD_REVISIONS = {
    "CurbRamp": "969ad5a33079db7885c9e9a4eb5cd0f39d8fe57a",
    "NoCurbRamp": "9f3f3bb2f40864a468fb65e43644ee5ba54e3e19",
    "Obstacle": "8f0fe8fc48748fac35098e74c1848d015126deef",
    "SurfaceProblem": "0e1439daa5667c8a81f63e75f835bd51221df4c2",
    "Crosswalk": "b7936e5bd17a4b96db301f954e5d3a5fe9eaf361",
}
"""Model revisions sidewalk-ai-api's model_map.py pins (docs/calibration-2026-10.md, "Scope")."""

PACKET_METHODS = ("rollout", "cls_attention", "gradcam", "gradxact", "centre_blob")
"""Panels in the rater packet; ``gradcam``/``gradxact`` are the predicted class's maps; ``centre_blob`` is a control."""
PANEL_LETTERS = "ABCDE"

# Coding-sheet vocabulary (the rubric in docs/explanations-2026-10.md explains each value).
COVERS = ("yes", "partly", "no", "unclear")
COVERS_ORDER = ("no", "partly", "yes")
BEST = (*PANEL_LETTERS, "none")
SHEET_COLUMNS = ["packet_id", "label_point_uncertain", "rater",
                 *[f"covers_{p}" for p in PANEL_LETTERS], "best_panel", "notes"]
KEY_COLUMNS = ["packet_id", "overlay_id", "label_type", "category", "label_uid", "label_point_uncertain",
               *[f"panel_{p}" for p in PANEL_LETTERS]]

# Go/no-go rule (docs/explanations-2026-10.md, "Go/no-go"): thresholds applied by ``go_no_go``.
GO_MIN_RATE = 0.70
GO_MIN_MARGIN_OVER_BLOB = 0.20
GO_MIN_KAPPA = 0.60
GO_MAX_PREVALENCE = 0.90
GO_MIN_PERCENT_AGREEMENT = 0.85
GO_MIN_ITEMS = 20


def default_checkpoint(label_type):
    """The production model repo for ``label_type`` (``projectsidewalk/sidewalk-validator-ai-crosswalk``)."""
    return f"projectsidewalk/sidewalk-validator-ai-{label_type.lower()}"


# ---------------------------------------------------------------------------------------------------------------------
# Input, deployments and categories -- pure, unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def load_crops_dir(crops_dir, split="test"):
    """Manifest of ``<crops_dir>/<split>/<class>/<prefix>_<label_id>.<ext>`` (the Hugging Face dataset layout).

    Columns ``crop_id`` (the file stem), ``prefix``, ``label_id``, ``truth``, ``crop_path``. The prefix is the
    deployment's subdomain with ``-`` written as ``_``; ``public_deployments`` turns it into a ``city_id``.
    """
    rows = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(crops_dir, split, cls, "*"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            prefix, _, label_id = stem.rpartition("_")
            rows.append({"crop_id": stem, "prefix": prefix, "label_id": label_id, "truth": cls, "crop_path": path})
    if not rows:
        raise FileNotFoundError(f"no images under {crops_dir}/{split}/{{correct,incorrect}}/")
    df = pd.DataFrame(rows)
    if not df.crop_id.is_unique:
        raise ValueError(f"{crops_dir}/{split}: a crop appears in both class folders")
    return df


def load_split_csv(path, split="test"):
    """Manifest from a ``split_data.py`` ``split.csv`` (rows of ``split``); ``crop_path`` relative to the CSV's dir.

    ``prefix`` is the ``city`` column, which ``public_deployments`` accepts as a public ``city_id`` or subdomain.
    """
    df = pd.read_csv(path)
    df = df[df.split == split].copy()
    if not df.label_uid.is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    base = os.path.dirname(os.path.abspath(path))
    df["crop_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df.crop_path]
    df["crop_id"] = df.label_uid
    df["prefix"] = df.city
    return df.rename(columns={"label_class": "truth"})[["crop_id", "prefix", "label_id", "truth", "crop_path"]]


DATASET_CITY_ALIASES = {"sea": "seattle-wa"}
"""Dataset prefixes known to belong to a deployment, used only when that deployment is public in the listing."""


def public_deployments(prefixes, listing):
    """``{prefix: city_id}`` for the prefixes that **exactly** name a public deployment in a ``/v3/api/cities`` listing.

    A prefix matches when it equals a public deployment's subdomain (``sidewalk-<subdomain>.cs...``, from
    ``data_generation.discover_cities``) with ``-`` written as ``_``, equals a public ``city_id``, or is in
    ``DATASET_CITY_ALIASES``. There is no prefix-of matching. A prefix that also equals a non-public deployment's
    subdomain or ``city_id`` is dropped, and so is anything unrecognised: callers drop unmapped rows, so the filter
    fails closed.

    >>> listing = {"cities": [
    ...     {"city_id": "springfield-xx", "visibility": "public", "url": "https://sidewalk-springfield.cs.washington.edu"},
    ...     {"city_id": "shelbyville-xx", "visibility": "private", "url": "https://sidewalk-shelbyville.cs.washington.edu"}]}
    >>> public_deployments(["springfield", "shelbyville", "spring", "springfield-xx"], listing)
    {'springfield': 'springfield-xx', 'springfield-xx': 'springfield-xx'}
    """
    from data_generation import discover_cities, listing_entries, subdomain_from_url

    if isinstance(listing, str):
        with open(listing) as f:
            listing = json.load(f)
    public = discover_cities(listing, public_only=True)                    # {city_id: subdomain}
    by_name = {sub.replace("-", "_"): cid for cid, sub in public.items()}
    by_name.update({cid: cid for cid in public})
    for alias, cid in DATASET_CITY_ALIASES.items():
        if cid in public:
            by_name.setdefault(alias, cid)
    blocked = set()
    for c in listing_entries(listing):
        if c.get("visibility") == "public":
            continue
        blocked.add(c["city_id"])
        blocked.add(c["city_id"].replace("-", "_"))
        url = (c.get("url") or "").strip()
        if url:
            blocked.add(subdomain_from_url(url).replace("-", "_"))
    return {p: by_name[p] for p in prefixes if p in by_name and p not in blocked}


def label_point_uncertain(sizes):
    """True for crops that are not square: the 2025 crops were clipped at the view border, moving the label point.

    >>> label_point_uncertain([(640, 640), (640, 323)])
    [False, True]
    """
    return [w != h for w, h in sizes]


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
    Within a category the sample is ordered by descending confidence, and ``overlay_id`` is ``<type>-<cat>-<NN>``
    with ``tile`` = NN, the number printed on the contact-sheet tile.

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
# Maps and map statistics -- rollout and the statistics are pure numpy; the gradient maps need torch
# ---------------------------------------------------------------------------------------------------------------------

def discard_lowest(fused, ratio):
    """Zero the smallest ``ratio`` share of a fused attention matrix's entries, never the CLS->CLS entry ``[0, 0]``.

    >>> discard_lowest(np.array([[0.1, 0.9], [0.6, 0.4]]), 0.5).tolist()
    [[0.1, 0.9], [0.6, 0.0]]
    """
    flat = np.asarray(fused, dtype=np.float64).reshape(-1).copy()
    k = int(flat.size * ratio)
    order = np.argsort(flat, kind="stable")[:k]
    order = order[order != 0]
    flat[order] = 0
    return flat.reshape(np.shape(fused))


def attention_rollout(attentions, head_fusion="mean", discard_ratio=0.0, num_prefix_tokens=1):
    """Attention rollout: the CLS token's rolled-out attention to each patch token.

    Args:
        attentions: one array per layer (first layer first), each ``(heads, T, T)`` or ``(1, heads, T, T)``, rows
            summing to 1, token 0 being CLS.
        head_fusion: ``mean``, ``max`` or ``min`` over heads.
        discard_ratio: share of the smallest fused entries per layer to zero first (``discard_lowest``).
        num_prefix_tokens: CLS (+ register) tokens before the patches; DINOv2 without registers has 1.

    Returns:
        ``(T - num_prefix_tokens,)``: row 0 of ``R_L @ ... @ R_1`` over the patch columns, where
        ``R_l = rownorm((A_l + I) / 2)``.

    Example (hand-computed; layer 2 is applied *after* layer 1, and it is the CLS *row*)::

        >>> a1 = np.array([[[1., 0, 0], [1, 0, 0], [0, 0, 1]]])   # R1 = [[1,0,0],[.5,.5,0],[0,0,1]]
        >>> a2 = np.array([[[0., 1, 0], [0, 1, 0], [0, 1, 0]]])   # R2 = [[.5,.5,0],[0,1,0],[0,.5,.5]]
        >>> attention_rollout([a1, a2]).tolist()                  # row 0 of R2 @ R1 = [.75, .25, 0]
        [0.25, 0.0]
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
        if discard_ratio > 0:
            fused = discard_lowest(fused, discard_ratio)
        t = fused.shape[-1]
        a = 0.5 * fused + 0.5 * np.eye(t)
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


def centre_blob(n=16, sigma=2.5):
    """A Gaussian at the centre of an ``n x n`` grid: the control map in the rater packet."""
    c = (n - 1) / 2
    y, x = np.mgrid[0:n, 0:n]
    return np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))


def centre_mass(grid, frac=0.5):
    """Share of a non-negative map's total in the central ``frac`` x ``frac`` box of the grid (uniform: ``frac**2``).

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


def border_mass(grid):
    """Share of a non-negative map's total on the outermost ring of patches (uniform 16 x 16: 60/256 = 0.234).

    >>> border_mass(np.ones((4, 4)))
    0.75
    """
    g = np.clip(np.asarray(grid, dtype=np.float64), 0, None)
    total = g.sum()
    if total <= 0:
        return float("nan")
    return float((total - g[1:-1, 1:-1].sum()) / total)


def top_mass(values, k=3):
    """Share of a non-negative map's total held by its ``k`` largest patches (uniform 16 x 16: 3/256 = 0.012).

    >>> top_mass([1, 1, 1, 1], k=1)
    0.25
    """
    v = np.sort(np.clip(np.asarray(values, dtype=np.float64).ravel(), 0, None))[::-1]
    return float(v[:k].sum() / v.sum()) if v.sum() > 0 else float("nan")


def top_overlap(values, reference, k=3):
    """Share of ``values``' top-``k`` patches that are in ``reference`` (a set of patch indices).

    >>> top_overlap([0, 5, 1, 4], {1, 2}, k=2)
    0.5
    """
    top = np.argsort(np.asarray(values, dtype=np.float64).ravel(), kind="stable")[::-1][:k]
    return float(np.mean([int(i) in set(reference) for i in top]))


def high_norm_tokens(layer_norms, k=5):
    """``(layer, top-k patch indices, max/median ratio)`` of the layer with the largest max/median token-norm ratio.

    ``layer_norms`` is ``(layers, patches)`` (from ``explain_one``). Large DINOv2 models put a few tokens with very
    high norm on low-information patches (Darcet et al. 2024, "Vision Transformers Need Registers").

    >>> high_norm_tokens([[1, 1, 1, 1], [1, 9, 1, 2]], k=1)
    (1, {1}, 6.0)
    """
    n = np.asarray(layer_norms, dtype=np.float64)
    ratio = n.max(1) / np.median(n, 1)
    layer = int(ratio.argmax())
    return layer, set(int(i) for i in np.argsort(n[layer], kind="stable")[::-1][:k]), float(ratio[layer])


def map_statistics(maps, uncertain, highnorm):
    """Per-map ``centre``/``border``/``top3`` mass (NaN for centre/border when ``uncertain``) and the high-norm overlap.

    ``maps`` is ``{name: patch vector}``; ``highnorm`` a set of patch indices. Keys are ``<map>_<stat>``.
    """
    out = {}
    for name, v in maps.items():
        g = to_grid(v)
        out[f"{name}_centre"] = float("nan") if uncertain else centre_mass(g)
        out[f"{name}_border"] = float("nan") if uncertain else border_mass(g)
        out[f"{name}_top3"] = top_mass(v)
        if name in ("rollout", "cls_attention"):
            out[f"{name}_top3_in_highnorm"] = top_overlap(v, highnorm)
    return out


def gradcam_module(model):
    """The module whose output Grad-CAM uses: the last encoder block's ``norm1``."""
    return model.dinov2.encoder.layer[-1].norm1


def block_module(model):
    """The module whose output grad x act uses: the last encoder block (the residual stream after it)."""
    return model.dinov2.encoder.layer[-1]


def explain_one(model, pixel_values, target_class=None, num_prefix_tokens=1):
    """One forward pass and one backward pass per class: probabilities, the class-agnostic maps and gradient maps.

    ``model`` is a ``Dinov2ForImageClassification`` loaded with ``attn_implementation="eager"``; ``pixel_values`` is
    ``(1, 3, H, W)`` on the model's device. ``target_class`` (a logit index) names the "verdict" maps; it defaults to
    the argmax (the first index on a tie, as the API does).

    Returns:
        dict with ``probs`` (per logit index), ``target``, ``rollout``, ``cls_attention`` (heads x patches),
        ``gradcam``/``gradxact`` (``{logit index: patch vector}``), ``gradcam_target``/``gradxact_target`` and
        ``layer_norms`` (layers x patches: each block's output patch-token norms).
    """
    import torch

    acts, norms = {}, []

    def keep(name):
        def hook(_module, _inputs, output):
            acts[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def keep_norm(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        norms.append(h[0, num_prefix_tokens:].detach().norm(dim=-1).float().cpu().numpy())

    handles = [gradcam_module(model).register_forward_hook(keep("norm1")),
               block_module(model).register_forward_hook(keep("block"))]
    handles += [layer.register_forward_hook(keep_norm) for layer in model.dinov2.encoder.layer]
    try:
        with torch.enable_grad():
            out = model(pixel_values=pixel_values, output_attentions=True)
            if out.attentions is None or out.attentions[0] is None:
                raise RuntimeError('no attention weights: load the model with attn_implementation="eager"')
            logits = out.logits[0]
            n_cls = logits.numel()
            target = int(logits.argmax()) if target_class is None else int(target_class)
            gradcam, gradxact = {}, {}
            for c in range(n_cls):
                others = torch.cat([logits[:c], logits[c + 1:]])
                score = logits[c] - (others[0] if n_cls == 2 else torch.logsumexp(others, 0))
                g_n, g_b = torch.autograd.grad(score, [acts["norm1"], acts["block"]], retain_graph=c < n_cls - 1)
                a_n = acts["norm1"][0, num_prefix_tokens:].detach()
                a_b = acts["block"][0, num_prefix_tokens:].detach()
                weights = g_n[0, num_prefix_tokens:].mean(dim=0)
                gradcam[c] = torch.relu(a_n @ weights).float().cpu().numpy()
                gradxact[c] = torch.relu((a_b * g_b[0, num_prefix_tokens:]).sum(-1)).float().cpu().numpy()
        attentions = [a.detach().float().cpu().numpy() for a in out.attentions]
        probs = logits.detach().softmax(-1).float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    return {"probs": probs, "target": target,
            "rollout": attention_rollout(attentions, num_prefix_tokens=num_prefix_tokens),
            "cls_attention": last_layer_cls_attention(attentions, num_prefix_tokens),
            "gradcam": gradcam, "gradxact": gradxact,
            "gradcam_target": gradcam[target], "gradxact_target": gradxact[target],
            "layer_norms": np.stack(norms)}


def grad_cam(model, pixel_values, target_class=None):
    """Grad-CAM per patch for ``target_class`` (see ``explain_one``); returns ``(cam, target_index)``."""
    r = explain_one(model, pixel_values, target_class)
    return r["gradcam_target"], r["target"]


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


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 10.1 has no scalable default font
        return ImageFont.load_default()


def mark_centre(image, uncertain=False, size=6, colour=(0, 255, 255)):
    """Copy of ``image`` with a small cross at the centre (the label point), or a yellow "?" tag when uncertain."""
    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    if uncertain:
        d.rectangle([(0, 0), (13, 13)], fill=(255, 255, 0))
        d.text((4, 0), "?", fill="black", font=_font(12))
        return np.asarray(im)
    cx, cy = im.width // 2, im.height // 2
    d.line([(cx - size, cy), (cx + size, cy)], fill=colour, width=1)
    d.line([(cx, cy - size), (cx, cy + size)], fill=colour, width=1)
    return np.asarray(im)


def strip(image, maps, uncertain=False):
    """Side-by-side ``crop | overlay(map) ...`` as one uint8 array; ``maps`` are patch grids."""
    panels = [mark_centre(image, uncertain)] + [heat_overlay(image, m) for m in maps]
    return np.concatenate(panels, axis=1)


def labelled_panels(image, maps, labels, uncertain=False, title=None):
    """Full-resolution packet image: the crop, then one overlay per map, each with its label above it."""
    panels = [mark_centre(image, uncertain)] + [heat_overlay(image, m) for m in maps]
    names = ["crop" + (" (label point uncertain)" if uncertain else " (+ = label point)")] + list(labels)
    h, w = image.shape[:2]
    cap, pad, head = 18, 6, (22 if title else 0)
    out = Image.new("RGB", (len(panels) * (w + pad) + pad, head + cap + h + pad), "white")
    d = ImageDraw.Draw(out)
    if title:
        d.text((pad, 3), title, fill="black", font=_font(14))
    for i, (p, n) in enumerate(zip(panels, names)):
        x = pad + i * (w + pad)
        d.text((x, head + 2), n, fill="black", font=_font(12))
        out.paste(Image.fromarray(p), (x, head + cap))
    return out


def contact_sheet(strips, captions, path, ncols=4, tile_height=112, header=None, quality=82):
    """Grid of downscaled strips with a caption above each; JPEG (photographs compress ~10x better than PNG)."""
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
# Blinded packet, coding and agreement -- pure, unit-tested
# ---------------------------------------------------------------------------------------------------------------------

def packet_key(samples, seed):
    """Blinding key: a random ``packet_id`` per sampled overlay, a shuffled order, and a random panel order.

    ``samples`` needs ``overlay_id, label_type, category, label_uid, label_point_uncertain``. Returns
    ``KEY_COLUMNS`` in packet order; ``panel_<letter>`` names the method shown in that panel. Deterministic in
    ``seed`` and the set of overlay ids, so the seed must stay secret until the rating is done: with it and the
    committed ``*_samples.csv`` anyone can rebuild the key.
    """
    rng = np.random.default_rng(seed)
    s = samples.sort_values("overlay_id").reset_index(drop=True)
    ids = set()
    while len(ids) < len(s):
        ids.add(f"p{int(rng.integers(0, 16 ** 6)):06x}")
    s = s.assign(packet_id=rng.permutation(sorted(ids)))
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    for letter in PANEL_LETTERS:
        s[f"panel_{letter}"] = ""
    for i in range(len(s)):
        for letter, method in zip(PANEL_LETTERS, rng.permutation(PACKET_METHODS)):
            s.loc[i, f"panel_{letter}"] = method
    return s[KEY_COLUMNS]


def blank_sheet(key, uncertain):
    """Blank rater sheet in packet order: ``packet_id``, ``label_point_uncertain`` and empty coding columns."""
    sheet = pd.DataFrame({"packet_id": key.packet_id.values,
                          "label_point_uncertain": [bool(uncertain[o]) for o in key.overlay_id]})
    for c in SHEET_COLUMNS[2:]:
        sheet[c] = ""
    return sheet[SHEET_COLUMNS]


def validate_codes(sheet):
    """Raise if a filled sheet uses a value outside the rubric's vocabulary (blank cells are allowed)."""
    allowed = {**{f"covers_{p}": COVERS for p in PANEL_LETTERS}, "best_panel": BEST}
    for col, values in allowed.items():
        bad = set(sheet[col].fillna("").astype(str).str.strip()) - set(values) - {""}
        if bad:
            raise ValueError(f"{col}: values {sorted(bad)} are not in the rubric ({', '.join(values)})")


def unblind(sheet, key):
    """``(long, best)``: the filled sheet joined to its key, one row per panel with its method, and the best panel.

    ``long`` has ``packet_id, overlay_id, label_type, category, label_point_uncertain, method, covers``, the
    uncertainty flag coming from the key (a rater's sheet is editable); ``best`` has ``best_method``.
    """
    m = sheet.drop(columns=["label_point_uncertain"], errors="ignore").merge(key, on="packet_id",
                                                                              validate="one_to_one", how="inner")
    if len(m) != len(sheet):
        raise ValueError("sheet has packet ids that are not in the key")
    rows = []
    for letter in PANEL_LETTERS:
        rows.append(pd.DataFrame({"packet_id": m.packet_id, "overlay_id": m.overlay_id, "label_type": m.label_type,
                                  "category": m.category,
                                  "label_point_uncertain": m.label_point_uncertain.astype(str).str.lower() == "true",
                                  "method": m[f"panel_{letter}"],
                                  "covers": m[f"covers_{letter}"].fillna("").astype(str).str.strip()}))
    best = m.best_panel.fillna("").astype(str).str.strip()
    best_method = [m.loc[i, f"panel_{b}"] if b and b in tuple(PANEL_LETTERS) else b for i, b in zip(m.index, best)]
    return (pd.concat(rows, ignore_index=True),
            pd.DataFrame({"packet_id": m.packet_id, "label_type": m.label_type, "category": m.category,
                          "best_method": best_method}))


def kappa_row(x, y, ordinal=None):
    """``n, percent_agreement, kappa, weighted_kappa`` for two raters' codes (lists of equal length).

    Kappa is NaN when it is undefined (no codes, or both raters used one and the same value throughout); percent
    agreement then carries the result. ``ordinal`` (low to high) adds a linear-weighted kappa.

    Example (hand-computed with linear weights over no < partly < yes, disagreement weight |i - j| / 2: observed
    weighted disagreement (0.5 + 0 + 0 + 1) / 4 = 0.375; expected from the marginals 0.4375; kappa = 1 - 0.375 / 0.4375
    = 1/7)::

        >>> r = kappa_row(["no", "partly", "yes", "yes"], ["partly", "partly", "yes", "no"], ("no", "partly", "yes"))
        >>> round(r["weighted_kappa"], 6)
        0.142857
    """
    from sklearn.metrics import cohen_kappa_score

    n = len(x)
    row = {"n": n, "percent_agreement": float(np.mean([i == j for i, j in zip(x, y)])) if n else np.nan,
           "kappa": np.nan, "weighted_kappa": np.nan}
    if n and len(set(x) | set(y)) > 1:
        row["kappa"] = float(cohen_kappa_score(x, y))
        if ordinal:
            order = {v: i for i, v in enumerate(ordinal)}
            row["weighted_kappa"] = float(cohen_kappa_score([order[v] for v in x], [order[v] for v in y],
                                                            labels=list(range(len(ordinal))), weights="linear"))
    return row


def agreement(long1, long2, by=("method",), column="covers", ordinal=COVERS_ORDER):
    """Kappa between two raters' unblinded long frames, per ``by`` group.

    Rows join on ``(overlay_id, method)`` one-to-one; a row counts only when both raters coded it and neither said
    ``unclear``. Returns ``*by, n, percent_agreement, kappa, weighted_kappa``.

    Example (hand-computed: 3 of 4 agree, p_o = 0.75; rater 1 says yes 2/4, rater 2 3/4, p_e = 0.5; kappa = 0.5)::

        >>> a = pd.DataFrame({"overlay_id": list("abcd"), "method": "m", "covers": ["yes", "yes", "no", "no"]})
        >>> b = a.assign(covers=["yes", "yes", "no", "yes"])
        >>> float(agreement(a, b).kappa.iloc[0])
        0.5
    """
    m = long1.merge(long2, on=["overlay_id", "method"], suffixes=("_1", "_2"), validate="one_to_one")
    a, b = m[f"{column}_1"].fillna("").astype(str), m[f"{column}_2"].fillna("").astype(str)
    m = m[(a != "") & (b != "") & (a != "unclear") & (b != "unclear")]
    keys = [k if k in m.columns else f"{k}_1" for k in by]
    rows = []
    for g, part in m.groupby(keys, sort=True):
        g = g if isinstance(g, tuple) else (g,)
        rows.append({**dict(zip(by, g)), **kappa_row(part[f"{column}_1"].tolist(), part[f"{column}_2"].tolist(),
                                                      ordinal)})
    return pd.DataFrame(rows, columns=[*by, "n", "percent_agreement", "kappa", "weighted_kappa"])


def covers_rate(long, categories=AGREE_CATEGORIES):
    """Share of ``yes``/``partly`` among ``yes``/``partly``/``no`` codes, per label type and method.

    ``unclear`` and blank are excluded from the denominator; only ``categories`` (default the Agree verdicts) count.
    """
    sub = long[long.category.isin(categories) & long.covers.isin(COVERS_ORDER)]
    g = sub.groupby(["label_type", "method"])
    return pd.DataFrame({"n": g.size(), "rate": g.covers.apply(lambda s: s.isin(("yes", "partly")).mean())}
                        ).reset_index()


def reliability_test(x, y):
    """``n, percent_agreement, kappa, prevalence, test, reliable`` for two raters' covers codes on the same items.

    ``test`` is ``kappa`` when Cohen's kappa is defined and the most common code's share among both raters' codes
    (``prevalence``) is <= ``GO_MAX_PREVALENCE``; reliable then means kappa >= ``GO_MIN_KAPPA``. Otherwise kappa is
    undefined or distorted by prevalence (both raters saying ``yes`` to nearly everything gives kappa near 0 even at
    97% agreement), and reliable means percent agreement >= ``GO_MIN_PERCENT_AGREEMENT``.

    >>> reliability_test(["yes"] * 40, ["yes"] * 39 + ["no"])["test"], reliability_test(["yes"] * 40, ["yes"] * 39 + ["no"])["reliable"]
    ('percent_agreement', True)
    """
    r = kappa_row(list(x), list(y))
    prevalence = float(pd.Series(list(x) + list(y)).value_counts(normalize=True).max()) if r["n"] else np.nan
    if np.isfinite(r["kappa"]) and prevalence <= GO_MAX_PREVALENCE:
        test, reliable = "kappa", r["kappa"] >= GO_MIN_KAPPA
    else:
        test, reliable = "percent_agreement", bool(r["n"]) and r["percent_agreement"] >= GO_MIN_PERCENT_AGREEMENT
    return {"n": r["n"], "percent_agreement": r["percent_agreement"], "kappa": r["kappa"], "prevalence": prevalence,
            "test": test, "reliable": bool(reliable)}


def _agree_pairs(long1, long2):
    """Agree-verdict rows that both raters coded ``yes``/``partly``/``no``, joined on ``(overlay_id, method)``."""
    a = long1[long1.category.isin(AGREE_CATEGORIES)]
    b = long2[long2.category.isin(AGREE_CATEGORIES)]
    m = a.merge(b, on=["overlay_id", "method"], suffixes=("_1", "_2"), validate="one_to_one")
    return m[m.covers_1.isin(COVERS_ORDER) & m.covers_2.isin(COVERS_ORDER)]


RELIABILITY_COLUMNS = ["n", "percent_agreement", "kappa", "prevalence", "test", "reliable"]


def method_reliability(long1, long2):
    """``reliability_test`` per label type and method, on Agree-verdict items: the go/no-go rule's reliability gate."""
    m = _agree_pairs(long1, long2)
    rows = [{"label_type": lt, "method": meth, **reliability_test(part.covers_1, part.covers_2)}
            for (lt, meth), part in m.groupby(["label_type_1", "method"], sort=True)]
    return pd.DataFrame(rows, columns=["label_type", "method", *RELIABILITY_COLUMNS])


def pooled_reliability(long1, long2):
    """``reliability_test`` per label type over all panels pooled (reported for context; not part of the rule).

    Pooling cannot gate a single method: agreement on uninformative panels would lend reliability to a map the
    raters disagree on.
    """
    m = _agree_pairs(long1, long2)
    rows = [{"label_type": lt, **reliability_test(part.covers_1, part.covers_2)}
            for lt, part in m.groupby("label_type_1", sort=True)]
    return pd.DataFrame(rows, columns=["label_type", *RELIABILITY_COLUMNS])


def go_no_go(long1, long2, methods=("gradcam", "gradxact")):
    """Apply the doc's proposed go/no-go rule per label type; one row per type and candidate method.

    Rows with an uncertain label point must already be removed. On that type's Agree-verdict items (TP + FP) a
    method is **go** when all four hold:

    1. each rater coded at least ``GO_MIN_ITEMS`` of them ``yes``/``partly``/``no`` for this method;
    2. the covers rate ((yes + partly) / (yes + partly + no)), averaged over the two raters, is >= ``GO_MIN_RATE``;
    3. that rate exceeds the ``centre_blob`` control's averaged rate by >= ``GO_MIN_MARGIN_OVER_BLOB``;
    4. the raters are reliable on **this method's** codes (``reliability_test``: kappa >= ``GO_MIN_KAPPA``, or, when
       kappa is undefined or prevalence > ``GO_MAX_PREVALENCE``, percent agreement >= ``GO_MIN_PERCENT_AGREEMENT``).
    """
    r1, r2 = covers_rate(long1), covers_rate(long2)
    rates = r1.merge(r2, on=["label_type", "method"], suffixes=("_1", "_2"), how="outer", validate="one_to_one")
    rates["rate"] = rates[["rate_1", "rate_2"]].mean(axis=1)
    rel = method_reliability(long1, long2).set_index(["label_type", "method"])
    out = []
    for lt in sorted(rates.label_type.unique()):
        blob = rates[(rates.label_type == lt) & (rates.method == "centre_blob")].rate
        blob = float(blob.iloc[0]) if len(blob) else np.nan
        for meth in methods:
            r = rates[(rates.label_type == lt) & (rates.method == meth)]
            rate = float(r.rate.iloc[0]) if len(r) else np.nan
            n_min = int(min(np.nan_to_num(r.n_1.iloc[0]), np.nan_to_num(r.n_2.iloc[0]))) if len(r) else 0
            reliable = bool(rel.loc[(lt, meth), "reliable"]) if (lt, meth) in rel.index else False
            go = bool(n_min >= GO_MIN_ITEMS and rate >= GO_MIN_RATE and rate - blob >= GO_MIN_MARGIN_OVER_BLOB
                      and reliable)
            out.append({"label_type": lt, "method": meth, "n_min": n_min, "rate": rate, "centre_blob_rate": blob,
                        "reliable": reliable, "decision": "go" if go else "no-go"})
    return pd.DataFrame(out, columns=["label_type", "method", "n_min", "rate", "centre_blob_rate", "reliable",
                                      "decision"])


def planted_check(positions=(6, 15), seeds=20, hidden=32, image=56):
    """Do Grad-CAM and grad x act find the only patch token a classifier head reads? (needs torch + transformers)

    Builds a tiny random 2-layer Dinov2 (``image / 14`` squared patches) whose head is a linear layer on one patch
    token of the final hidden state, and counts over ``seeds`` random inputs how often each map's argmax is that
    token, for the predicted class's map and the other class's map. Chance is ``seeds / n_patches``.

    Returns:
        ``position, method, target, hits, n``.
    """
    import types
    import torch
    import transformers

    cfg = transformers.Dinov2Config(hidden_size=hidden, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2,
                                    image_size=image, patch_size=14, num_labels=2,
                                    id2label={0: "correct", 1: "incorrect"}, label2id={"correct": 0, "incorrect": 1})
    rows = []
    for k in positions:
        torch.manual_seed(0)
        base = transformers.Dinov2ForImageClassification._from_config(cfg, attn_implementation="eager").eval()
        lin = torch.nn.Linear(hidden, 2)

        class OneToken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dinov2, self.config, self.lin = base.dinov2, base.config, lin

            def forward(self, pixel_values, output_attentions=False, _k=k):
                o = self.dinov2(pixel_values=pixel_values, output_attentions=output_attentions)
                return types.SimpleNamespace(logits=self.lin(o.last_hidden_state[:, 1 + _k]), attentions=o.attentions)

        model = OneToken().eval()
        hits = {(m, t): 0 for m in ("gradcam", "gradxact") for t in ("predicted", "other")}
        for seed in range(seeds):
            torch.manual_seed(seed)
            r = explain_one(model, torch.randn(1, 3, image, image))
            for m in ("gradcam", "gradxact"):
                hits[(m, "predicted")] += int(np.argmax(r[m][r["target"]]) == k)
                hits[(m, "other")] += int(np.argmax(r[m][1 - r["target"]]) == k)
        rows += [{"position": k, "method": m, "target": t, "hits": h, "n": seeds} for (m, t), h in hits.items()]
    return pd.DataFrame(rows, columns=["position", "method", "target", "hits", "n"])


# ---------------------------------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------------------------------

def summarise(samples, stats=None):
    """Per label type and category: n, n with a certain label point, and the median of each statistic column.

    Medians skip NaN, so centre/border statistics are over certain crops only.
    """
    stats = stats or [c for c in samples.columns if c.endswith(("_centre", "_border", "_top3", "_in_highnorm"))]
    g = samples.groupby(["label_type", "category"], sort=True)
    out = g.size().rename("n").to_frame()
    if "label_point_uncertain" in samples.columns:
        out["n_certain"] = g.label_point_uncertain.apply(lambda s: int((~s.astype(bool)).sum()))
    for c in stats:
        out[c] = g[c].median()
    return out.reset_index()


def markdown_table(df, floatfmt="{:.3f}"):
    """A pipe table of ``df`` (no tabulate dependency)."""
    cells = [[floatfmt.format(v) if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def run(args):
    import torch

    label_type = args.label_type
    checkpoint = args.checkpoint or default_checkpoint(label_type)
    revision = args.revision or (PROD_REVISIONS[label_type] if not args.checkpoint else None)
    os.makedirs(os.path.join(args.out_dir, "maps"), exist_ok=True)
    torch.manual_seed(args.seed)

    manifest = (load_split_csv(args.split_csv, args.split) if args.split_csv
                else load_crops_dir(args.crops_dir, args.split))
    sizes = [Image.open(p).size for p in manifest.crop_path]
    manifest["width"], manifest["height"] = [w for w, _ in sizes], [h for _, h in sizes]
    manifest["label_point_uncertain"] = label_point_uncertain(sizes)
    print(f"{label_type}: {len(manifest)} crops in split {args.split}; model {checkpoint} @ {revision}; "
          f"device {args.device}")
    processor, model = load_model(checkpoint, revision, args.device)
    probs = predict(model, processor, manifest.crop_path.tolist(), args.device, args.batch_size)
    preds = add_verdicts(manifest.assign(p_correct=probs[:, 0], p_incorrect=probs[:, 1]))
    preds.drop(columns=["crop_path"]).to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    counts = {c: int((preds.category == c).sum()) for c in CATEGORIES}

    if args.include_private:
        pool = preds.assign(label_uid=["dataset:" + c for c in preds.crop_id])
    else:
        mapping = public_deployments(sorted(preds.prefix.unique()), args.cities_json)
        pool = preds[preds.prefix.isin(mapping)].copy()
        pool["label_uid"] = [f"{mapping[p]}:{i}" for p, i in zip(pool.prefix, pool.label_id)]
        print(f"sampling from public deployments only: {len(pool)} of {len(preds)} crops")
    if not pool.label_uid.is_unique:
        raise ValueError("label_uid is not unique in the sampling pool")
    pool_counts = {c: int((pool.category == c).sum()) for c in CATEGORIES}
    info = {"label_type": label_type, "checkpoint": checkpoint, "revision": revision, "device": args.device,
            "split": args.split, "n": int(len(preds)), "categories": counts,
            "n_label_point_uncertain": int(preds.label_point_uncertain.sum()),
            "public_only": not args.include_private, "n_sampling_pool": int(len(pool)),
            "categories_sampling_pool": pool_counts,
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
    samples = sample_categories(pool, args.per_category, args.seed, categories, label_type)
    idx = class_index(model)
    names = {v: k for k, v in idx.items()}
    stats_rows, max_dp, seconds = [], 0.0, []
    for row in samples.itertuples():
        inputs = processor(Image.open(row.crop_path).convert("RGB"), return_tensors="pt").to(args.device)
        t0 = time.perf_counter()
        r = explain_one(model, inputs["pixel_values"], idx[row.pred])
        seconds.append(time.perf_counter() - t0)     # explain_one copies its outputs to the CPU, so this is synced
        max_dp = max(max_dp, abs(float(r["probs"][idx["correct"]]) - row.p_correct))
        image = denormalise(inputs["pixel_values"][0].cpu().numpy(), processor.image_mean, processor.image_std)
        maps = {"rollout": r["rollout"], "cls_attention": r["cls_attention"].mean(0)}
        for i, c in names.items():
            maps[f"gradcam_{c}"] = r["gradcam"][i]
            maps[f"gradxact_{c}"] = r["gradxact"][i]
        layer, highnorm, ratio = high_norm_tokens(r["layer_norms"])
        np.savez_compressed(os.path.join(args.out_dir, "maps", f"{row.overlay_id}.npz"), image=image,
                            label_point_uncertain=row.label_point_uncertain, pred=row.pred,
                            **{k: v.astype(np.float32) for k, v in maps.items()})
        stats = map_statistics(maps, row.label_point_uncertain, highnorm)
        stats.update({"highnorm_layer": layer, "highnorm_ratio": ratio})
        stats_rows.append(stats)
    samples = pd.concat([samples.reset_index(drop=True), pd.DataFrame(stats_rows)], axis=1)
    samples["contact_sheet"] = [f"{label_type.lower()}_{c}.jpg" for c in samples.category]
    info["explain_pass_max_abs_diff_p_correct"] = max_dp
    info["explain_seconds_per_crop_median"] = float(np.median(seconds)) if seconds else None
    info["n_sampled_label_point_uncertain"] = int(samples.label_point_uncertain.sum())
    cols = ["label_type", "overlay_id", "label_uid", "truth", "pred", "confidence", "p_correct", "category",
            "label_point_uncertain", "width", "height", "contact_sheet", "tile", *(stats_rows[0] if stats_rows else [])]
    samples.assign(label_type=label_type)[cols].to_csv(os.path.join(args.out_dir, "samples.csv"), index=False,
                                                         float_format="%.6f")
    with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
        json.dump(info, f, indent=2)
    write_sheets(args.out_dir, args.ncols, args.tile_height, args.quality)
    print(json.dumps(info, indent=2))


def write_sheets(out_dir, ncols=3, tile_height=96, quality=78):
    """Contact sheets ``<type>_<cat>.jpg`` from a ``run`` output (``samples.csv``, ``counts.json``, ``maps/``)."""
    samples = pd.read_csv(os.path.join(out_dir, "samples.csv"))
    with open(os.path.join(out_dir, "counts.json")) as f:
        info = json.load(f)
    label_type = info["label_type"]
    for cat, rows in samples.groupby("category", sort=False):
        items = []
        for row in rows.itertuples():
            z = np.load(os.path.join(out_dir, "maps", f"{row.overlay_id}.npz"))
            shown = [z["rollout"], z["cls_attention"], z[f"gradcam_{row.pred}"], z[f"gradxact_{row.pred}"]]
            items.append((strip(z["image"], [to_grid(m) for m in shown], bool(row.label_point_uncertain)),
                          f"{row.tile:02d} {row.label_uid} {row.pred[:3]} {row.confidence:.3f}"
                          + (" ?" if row.label_point_uncertain else "")))
        header = (f"{label_type} {cat} (n={len(items)} of {info['categories_sampling_pool'][cat]}): crop (+ label "
                  "point; ? = uncertain) | rollout | last-layer CLS attention | Grad-CAM (verdict) | grad x act (verdict)")
        contact_sheet([s for s, _ in items], [c for _, c in items],
                      os.path.join(out_dir, f"{label_type.lower()}_{cat}.jpg"), ncols=ncols, tile_height=tile_height,
                      header=header, quality=quality)


def sha256_of(path):
    """Hex sha256 of a file."""
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(cwd, *args):
    import subprocess

    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def inside_this_repo(path):
    """True if ``path`` would land in any checkout (main or worktree) of the repo that holds this script.

    Compares git's common directory of the path's nearest existing parent with this script's, so a sibling worktree
    counts. Without git, falls back to a path-prefix check against this checkout only.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.dirname(os.path.abspath(path))
    while not os.path.isdir(d):
        d = os.path.dirname(d)
    mine, theirs = _git(here, "rev-parse", "--git-common-dir"), _git(d, "rev-parse", "--git-common-dir")
    if mine is None:
        return os.path.abspath(path).startswith(here + os.sep)
    if theirs is None:
        return False
    return os.path.realpath(os.path.join(here, mine)) == os.path.realpath(os.path.join(d, theirs))


def build_packet(out_dirs, packet_dir, key_path, seed):
    """Blinded packet from ``run`` outputs: shuffled images with random ids and panels, a blank sheet, and the key.

    Writes the key and ``<key>.sha256``; commit only the hash until the rating is done. Neither the item images nor
    the sheet carry the category, crowd class, verdict, confidence, overlay id or label id.
    """
    per_dir = {d: pd.read_csv(os.path.join(d, "samples.csv")) for d in out_dirs}
    samples = pd.concat(per_dir.values(), ignore_index=True)
    folder = {o: d for d, s in per_dir.items() for o in s.overlay_id}
    uncertain = dict(zip(samples.overlay_id, samples.label_point_uncertain.astype(bool)))
    key = packet_key(samples, seed)
    os.makedirs(os.path.join(packet_dir, "items"), exist_ok=True)
    for i, k in enumerate(key.itertuples()):
        z = np.load(os.path.join(folder[k.overlay_id], "maps", f"{k.overlay_id}.npz"))
        pred = str(z["pred"])
        maps = {"rollout": z["rollout"], "cls_attention": z["cls_attention"], "gradcam": z[f"gradcam_{pred}"],
                "gradxact": z[f"gradxact_{pred}"], "centre_blob": centre_blob().ravel()}
        order = [getattr(k, f"panel_{p}") for p in PANEL_LETTERS]
        im = labelled_panels(z["image"], [to_grid(maps[m]) for m in order], [f"panel {p}" for p in PANEL_LETTERS],
                             uncertain[k.overlay_id], title=f"{i + 1:03d}  {k.packet_id}  ({k.label_type})")
        im.save(os.path.join(packet_dir, "items", f"{i + 1:03d}_{k.packet_id}.jpg"), quality=90)
    blank_sheet(key, uncertain).to_csv(os.path.join(packet_dir, "coding_sheet.csv"), index=False)
    key.to_csv(key_path, index=False)
    with open(key_path + ".sha256", "w") as f:
        f.write(f"{sha256_of(key_path)}  {os.path.basename(key_path)}\n")
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="predict a split, sample TP/FP/FN/TN, compute maps, write contact sheets")
    p.add_argument("--label-type", required=True, choices=LABEL_TYPES)
    p.add_argument("--checkpoint", default=None,
                   help="local checkpoint or HF model id (default: the production repo for --label-type)")
    p.add_argument("--revision", default=None,
                   help="HF revision for --checkpoint (default with no --checkpoint: the pinned production revision)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--crops-dir", help="directory in the HF dataset layout: <dir>/<split>/<class>/<prefix>_<id>.webp")
    src.add_argument("--split-csv", help="split_data.py split.csv (crop_path relative to its directory)")
    p.add_argument("--cities-json", default=None,
                   help="v3 API /v3/api/cities response (required unless --include-private): overlays are sampled "
                        "from public deployments only and label_uid is <city_id>:<label_id>")
    p.add_argument("--include-private", action="store_true",
                   help="do not filter by deployment (local debugging only; never commit or share the output)")
    p.add_argument("--split", default="test", help="which split to explain (default test)")
    p.add_argument("--per-category", type=int, default=50, help="maximum overlays per category (default 50)")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                   help="comma-separated subset of TP,FP,FN,TN (positive class = correct; default all four, TN "
                        "being the control)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    p.add_argument("--device", default="cpu", help="torch device for all passes: cpu or mps (default cpu)")
    p.add_argument("--check-device", default=None,
                   help="also predict on this device and report max |dp| and category flips against --device")
    p.add_argument("--batch-size", type=int, default=16, help="batch size of the prediction pass (default 16)")
    p.add_argument("--ncols", type=int, default=3, help="tiles per row in the contact sheets (default 3)")
    p.add_argument("--tile-height", type=int, default=96, help="contact-sheet tile height in px (default 96)")
    p.add_argument("--quality", type=int, default=78, help="contact-sheet JPEG quality (default 78)")
    p.add_argument("--out-dir", required=True, help="output directory for this label type")

    w = sub.add_parser("sheets", help="re-render the contact sheets of `run` outputs (no torch needed)")
    w.add_argument("out_dirs", nargs="+", help="`run` output directories")
    w.add_argument("--ncols", type=int, default=3, help="tiles per row (default 3)")
    w.add_argument("--tile-height", type=int, default=96, help="tile height in px (default 96)")
    w.add_argument("--quality", type=int, default=78, help="JPEG quality (default 78)")

    s = sub.add_parser("summary", help="per type x category medians of the map statistics in samples.csv files")
    s.add_argument("samples", nargs="+", help="samples.csv files written by `run`")
    s.add_argument("--stats", default=None, help="comma-separated statistic columns (default: all)")

    k = sub.add_parser("packet", help="blinded rater packet (keep it outside the repo) and its key")
    k.add_argument("out_dirs", nargs="+", help="`run` output directories")
    k.add_argument("--out-dir", required=True, help="packet directory (outside the repo)")
    k.add_argument("--key", required=True,
                   help="CSV for the blinding key, outside the repo (raters must not see it); <key>.sha256 is "
                        "written next to it and is the only part to commit before rating")
    k.add_argument("--seed", type=int, required=True,
                   help="shuffling seed; no default, keep it secret until rating is done (with it and the committed "
                        "samples anyone can rebuild the key)")

    a = sub.add_parser("agreement", help="unblind two raters' sheets; kappa per method and type; go/no-go")
    a.add_argument("sheet1")
    a.add_argument("sheet2")
    a.add_argument("--key", required=True, help="the packet key written by `packet`")
    a.add_argument("--key-sha256", default=None,
                   help="optional committed hash file (docs/explanations-2026-10/packet_key.sha256) to check --key "
                        "against")

    c = sub.add_parser("planted-check", help="Grad-CAM vs grad x act on a tiny model whose head reads one token")
    c.add_argument("--positions", default="6,15", help="patch tokens the head reads (default 6,15 of 16)")
    c.add_argument("--seeds", type=int, default=20, help="random inputs per position (default 20)")
    a.add_argument("--out", default=None, help="optional prefix for CSV copies of the tables")

    args = parser.parse_args(argv)
    if args.command == "run":
        if not args.cities_json and not args.include_private:
            parser.error("--cities-json is required (or --include-private, for local debugging only)")
        run(args)
    elif args.command == "sheets":
        for d in args.out_dirs:
            write_sheets(d, args.ncols, args.tile_height, args.quality)
    elif args.command == "summary":
        table = summarise(pd.concat([pd.read_csv(f) for f in args.samples], ignore_index=True),
                          args.stats.split(",") if args.stats else None)
        print(markdown_table(table))
    elif args.command == "planted-check":
        print(markdown_table(planted_check(tuple(int(x) for x in args.positions.split(",")), args.seeds)))
    elif args.command == "packet":
        if inside_this_repo(args.key):
            parser.error("--key must be outside every checkout of this repo until the rating is done "
                         "(commit only <key>.sha256)")
        key = build_packet(args.out_dirs, args.out_dir, args.key, args.seed)
        print(f"wrote {len(key)} items to {args.out_dir}/items, {args.out_dir}/coding_sheet.csv and {args.key}")
    else:
        if args.key_sha256:
            with open(args.key_sha256) as f:
                expected = f.read().split()[0]
            if sha256_of(args.key) != expected:
                raise SystemExit(f"{args.key} does not match {args.key_sha256}")
        key = pd.read_csv(args.key, dtype=str)
        longs = []
        for path in (args.sheet1, args.sheet2):
            sheet = pd.read_csv(path, dtype=str)
            validate_codes(sheet)
            long, _ = unblind(sheet, key)
            longs.append(long[~long.label_point_uncertain])
        tables = {"agreement_by_method": agreement(*longs, by=("method",)),
                  "agreement_by_type_method": agreement(*longs, by=("label_type", "method")),
                  "reliability_by_type_method": method_reliability(*longs),
                  "reliability_pooled_context": pooled_reliability(*longs),
                  "go_no_go": go_no_go(*longs)}
        for name, t in tables.items():
            print(f"== {name} ==")
            print(markdown_table(t))
            if args.out:
                t.to_csv(f"{args.out}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
