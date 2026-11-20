"""Backbone registry for the validator (issue #119): which pretrained image encoder ``fine_tune_classifier.py`` starts
from, how each one is loaded, and what is recorded about it.

Every validator since 2025-08 fine-tunes ``facebook/dinov2-large``. This module makes the backbone a recorded option
(``fine_tune_classifier.py --backbone <key>``) so other encoders can be compared on the round-3 data with everything
else held fixed (docs/backbone-2026-11.md). The default key, ``dinov2-large``, keeps the training path exactly as it
was: ``fine_tune_classifier.build_model`` still builds it, and its run key is unchanged.

What is held fixed across backbones: the slow ``BitImageProcessor`` of the baseline (shortest edge 256, bicubic, centre
crop 224, ImageNet mean and std), the train-time augmentation, and the recipe. What differs, and is recorded: the
patch grid (tokens per image), the pooling, and the parameter count (``backbone_settings``).

Families:

- ``dinov2``: ``Dinov2ForImageClassification`` (the baseline; head on [CLS, mean of patch tokens]).
- ``dinov2_with_registers``: ``Dinov2WithRegistersForImageClassification`` (same pooling, registers left out).
- ``convnextv2``: ``ConvNextV2ForImageClassification`` (global average pool and layernorm). RampNet's backbone
  (``projectsidewalk/rampnet-model``, a timm ConvNeXt V2-B) is converted into this class by ``convert-rampnet``.
- ``dinov3_vit``: transformers has no classification class for DINOv3, so ``_dinov3_classes()`` defines a small one
  (``SidewalkDinov3ForImageClassification``) with the baseline's pooling: [CLS, mean of patch tokens], the four
  register tokens left out. ``register_auto()`` registers it with ``AutoModelForImageClassification`` so
  ``evaluate.py`` and ``export_onnx.py`` load its checkpoints like any other. Its weights are gated on HF; a token
  with access is required to download them.

torch, transformers and huggingface_hub are imported inside functions, so this module imports on the CPU-only CI.

Every model ``build_classifier`` returns has ``accepts_loss_kwargs = False``: the Trainer then divides the loss by the
gradient-accumulation steps, as transformers 4.55.0 did for every round-3 run (under 4.56 the stock DINOv2 classes
would otherwise sum and divide by the items in the accumulated batch, and the ConvNeXt V2 class would fail on the
first step).

CLI::

    python backbones.py list                                     # the registry as a markdown table
    python backbones.py prefetch --backbones dinov2-large,dinov2-reg-large,convnextv2-base,rampnet-convnextv2-base \\
        --out $OUT/backbone_manifest.json                        # login node, network; gated rows need HF_TOKEN
    python backbones.py convert-rampnet --out $OUT/backbones/rampnet-convnextv2-base --verify-with-timm
    hyakstorage /gscratch/makelab | python backbones.py storage-free --min-gb 10 --min-files 2000

Example (Python)::

    import backbones
    processor = backbones.build_image_processor("dinov2-reg-large")
    model = backbones.build_classifier("dinov2-reg-large", {0: "correct", 1: "incorrect"},
                                       {"correct": 0, "incorrect": 1})
"""
import argparse
import functools
import hashlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RANDOM_INIT_ENV = "SVA_BACKBONE_RANDOM_INIT"
BACKBONE_DIR_ENV = "SVA_BACKBONE_DIR"
GATED_REASON = "weights are gated on HF; a token with access is required"
CONVERSION_FILE = "conversion.json"
# Arch of each family at the size the registry uses, for SVA_BACKBONE_RANDOM_INIT=1 (smoke tests without weights).
_DINOV2_L = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "mlp_ratio": 4,
             "image_size": 518, "patch_size": 14, "layerscale_value": 1.0}
_CONVNEXTV2_B = {"depths": [3, 3, 27, 3], "hidden_sizes": [128, 256, 512, 1024]}


def _dinov3_arch(hidden, layers, heads):
    return {"hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
            "intermediate_size": 4 * hidden, "num_register_tokens": 4, "layerscale_value": 1e-5,
            "patch_size": 16, "use_gated_mlp": False}


@dataclass(frozen=True)
class Backbone:
    """One registry row. ``key`` is the CLI name and the provenance value; ``source``/``revision`` the HF repo and
    the full commit it is pinned to; ``patch_size`` None for the convnet (stride 32); ``num_prefix_tokens`` the
    tokens before the patch tokens (CLS, plus registers); ``converted`` rows load from ``convert-rampnet`` output
    under the backbone directory instead of the hub; ``arch`` is the config used for a random-init smoke test."""
    key: str
    source: str
    revision: str
    family: str
    patch_size: int | None
    num_prefix_tokens: int
    params_m: float
    resize_px: int = 256
    crop_px: int = 224
    gated: bool = False
    converted: bool = False
    weights_file: str = "model.safetensors"
    arch: dict = field(default_factory=dict)

    @property
    def tokens_per_image(self):
        """Patch tokens the encoder sees at ``crop_px`` (prefix tokens not counted); the convnet's 7 x 7 map at 224.

        >>> get("dinov2-large").tokens_per_image, get("dinov3-vitl16").tokens_per_image, get("convnextv2-base").tokens_per_image
        (256, 196, 49)
        """
        stride = self.patch_size or 32
        return (self.crop_px // stride) ** 2


DEFAULT = "dinov2-large"
_ROWS = (
    Backbone("dinov2-large", "facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", "dinov2",
             14, 1, 304.4, arch=_DINOV2_L),
    Backbone("dinov2-reg-large", "facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
             "dinov2_with_registers", 14, 5, 304.4, arch={**_DINOV2_L, "num_register_tokens": 4}),
    Backbone("convnextv2-base", "facebook/convnextv2-base-22k-384", "53ec0af6e1bd11bf7eba3e9131d29b69625accf3",
             "convnextv2", None, 0, 88.7, arch=_CONVNEXTV2_B),
    Backbone("rampnet-convnextv2-base", "projectsidewalk/rampnet-model", "606a11956743f7eb328d9207769034752f6191f4",
             "convnextv2", None, 0, 88.7, converted=True, arch=_CONVNEXTV2_B),
    Backbone("rampnet-crop-convnextv2-base", "projectsidewalk/rampnet-crop-model",
             "7aa79b8edb10b384ed69c2e99f74945e9c527fd3", "convnextv2", None, 0, 88.7, converted=True,
             weights_file="round2_ps_and_manual_best_model.safetensors", arch=_CONVNEXTV2_B),
    Backbone("dinov3-vitl16", "facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51",
             "dinov3_vit", 16, 5, 303.1, gated=True, arch=_dinov3_arch(1024, 24, 16)),
    Backbone("dinov3-vitl16-256", "facebook/dinov3-vitl16-pretrain-lvd1689m",
             "ea8dc2863c51be0a264bab82070e3e8836b02d51", "dinov3_vit", 16, 5, 303.1, resize_px=292, crop_px=256,
             gated=True, arch=_dinov3_arch(1024, 24, 16)),
    Backbone("dinov3-vitb16", "facebook/dinov3-vitb16-pretrain-lvd1689m", "5931719e67bbdb9737e363e781fb0c67687896bc",
             "dinov3_vit", 16, 5, 85.7, gated=True, arch=_dinov3_arch(768, 12, 12)),
    Backbone("dinov3-vits16", "facebook/dinov3-vits16-pretrain-lvd1689m", "114c1379950215c8b35dfcd4e90a5c251dde0d32",
             "dinov3_vit", 16, 5, 21.6, gated=True, arch=_dinov3_arch(384, 12, 6)),
)
REGISTRY = {b.key: b for b in _ROWS}
# Arms allowed to win the comparison (docs/backbone-2026-11.md, the pre-registered rule); the others are explanatory.
ELIGIBLE = ("dinov2-reg-large", "rampnet-convnextv2-base", "dinov3-vitl16")
POOLING = {"dinov2": "concat(CLS, mean of patch tokens)",
           "dinov2_with_registers": "concat(CLS, mean of patch tokens), registers left out",
           "dinov3_vit": "concat(CLS, mean of patch tokens), registers left out",
           "convnextv2": "global average pool, then layernorm"}
MODEL_CLASS = {"dinov2": "Dinov2ForImageClassification",
               "dinov2_with_registers": "Dinov2WithRegistersForImageClassification",
               "dinov3_vit": "SidewalkDinov3ForImageClassification",
               "convnextv2": "ConvNextV2ForImageClassification"}


def get(key):
    """The registry row ``key``; ``KeyError`` naming the known keys otherwise.

    >>> get("convnextv2-base").family
    'convnextv2'
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown backbone {key!r}; known: {', '.join(REGISTRY)}")
    return REGISTRY[key]


def is_random_init(env=None):
    """True when ``SVA_BACKBONE_RANDOM_INIT=1`` (smoke tests only: weights are initialised randomly, no hub access).

    >>> is_random_init({"SVA_BACKBONE_RANDOM_INIT": "1"}), is_random_init({})
    (True, False)
    """
    env = os.environ if env is None else env
    return env.get(RANDOM_INIT_ENV, "") == "1"


# ---------------------------------------------------------------------------------------------------------------------
# Preprocessing (held fixed across backbones)
# ---------------------------------------------------------------------------------------------------------------------

def preprocess_spec(b):
    """The slow ``BitImageProcessor`` config for registry row ``b`` (a ``Backbone`` or a key): shortest edge
    ``resize_px`` bicubic, centre crop ``crop_px``, rescale 1/255, ImageNet mean and std, RGB. For ``dinov2-large``
    it equals the hub's ``preprocessor_config.json``.

    >>> s = preprocess_spec("dinov2-large")
    >>> s["size"], s["crop_size"], s["resample"]
    ({'shortest_edge': 256}, {'height': 224, 'width': 224}, 3)
    """
    b = get(b) if isinstance(b, str) else b
    return {"do_resize": True, "size": {"shortest_edge": b.resize_px}, "resample": 3,
            "do_center_crop": True, "crop_size": {"height": b.crop_px, "width": b.crop_px},
            "do_rescale": True, "rescale_factor": 1 / 255,
            "do_normalize": True, "image_mean": list(IMAGENET_MEAN), "image_std": list(IMAGENET_STD),
            "do_convert_rgb": True}


def preprocess_sha256(b):
    """sha256 of ``preprocess_spec(b)`` as sorted JSON (stable across runs and Python versions)."""
    return hashlib.sha256(json.dumps(preprocess_spec(b), sort_keys=True).encode()).hexdigest()


def build_image_processor(key):
    """The processor a run trains, evaluates and serves with. The default key keeps the call
    ``fine_tune_classifier.py`` always made (``AutoImageProcessor.from_pretrained("facebook/dinov2-large",
    use_fast=False)``); every other key builds ``BitImageProcessor(**preprocess_spec(key))`` with no hub access."""
    b = get(key)
    if key == DEFAULT:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(b.source, use_fast=False)
    from transformers import BitImageProcessor
    return BitImageProcessor(**preprocess_spec(b))


# ---------------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------------

def resolve_source(b, backbone_dir=None):
    """``(path or repo id, revision)`` to load row ``b`` from: the hub id and its pinned revision, or
    ``<backbone_dir>/<key>`` (revision None) for a converted row, which must hold ``config.json`` and
    ``conversion.json``."""
    b = get(b) if isinstance(b, str) else b
    if not b.converted:
        return b.source, b.revision
    backbone_dir = backbone_dir or os.environ.get(BACKBONE_DIR_ENV)
    if not backbone_dir:
        raise FileNotFoundError(f"{b.key} is converted: pass --backbone-dir (or set {BACKBONE_DIR_ENV}) to the "
                                f"directory holding {b.key}/ from `python backbones.py convert-rampnet`")
    path = os.path.join(backbone_dir, b.key)
    for name in ("config.json", CONVERSION_FILE):
        if not os.path.isfile(os.path.join(path, name)):
            raise FileNotFoundError(f"{path}/{name} not found; build it with `python backbones.py convert-rampnet "
                                    f"--out {path}`")
    return path, None


@functools.lru_cache(maxsize=1)
def _dinov3_classes():
    """``(SidewalkDinov3Config, SidewalkDinov3ForImageClassification)``, built once (transformers >= 4.56).

    The classifier reads ``[CLS, mean of patch tokens]`` of ``DINOv3ViTModel``'s last hidden state, so the four
    register tokens never reach the head, as ``Dinov2WithRegistersForImageClassification`` does. Its ``forward``
    takes no ``**kwargs`` and ``accepts_loss_kwargs`` is False: the Trainer divides the loss by the accumulation
    steps."""
    import torch
    from torch import nn
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from transformers.modeling_outputs import ImageClassifierOutput
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTPreTrainedModel

    class SidewalkDinov3Config(DINOv3ViTConfig):
        model_type = "sidewalk_dinov3_classifier"

    class SidewalkDinov3ForImageClassification(DINOv3ViTPreTrainedModel):
        config_class = SidewalkDinov3Config
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.dinov3_vit = DINOv3ViTModel(config)   # the attribute name must equal base_model_prefix
            self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
            self.post_init()

        def forward(self, pixel_values=None, labels=None):
            seq = self.dinov3_vit(pixel_values=pixel_values).last_hidden_state
            n = 1 + self.config.num_register_tokens
            pooled = torch.cat([seq[:, 0], seq[:, n:].mean(dim=1)], dim=1)
            logits = self.classifier(pooled)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))
            return ImageClassifierOutput(loss=loss, logits=logits)

    return SidewalkDinov3Config, SidewalkDinov3ForImageClassification


def has_dinov3():
    """True when this transformers has DINOv3 (4.56 and later)."""
    try:
        import transformers
        return hasattr(transformers, "DINOv3ViTModel")
    except ImportError:
        return False


_REGISTERED = []


def register_auto():
    """Register the DINOv3 classifier with ``AutoConfig`` / ``AutoModelForImageClassification`` so a saved checkpoint
    loads through ``AutoModelForImageClassification.from_pretrained``. Idempotent; returns False (no-op) when this
    transformers has no DINOv3 or is not installed, True otherwise."""
    if _REGISTERED:
        return True
    if not has_dinov3():
        return False
    from transformers import AutoConfig, AutoModelForImageClassification
    cfg_cls, model_cls = _dinov3_classes()
    try:
        AutoConfig.register(cfg_cls.model_type, cfg_cls)
        AutoModelForImageClassification.register(cfg_cls, model_cls)
    except ValueError:   # registered by an earlier import of this module under another name
        pass
    _REGISTERED.append(True)
    return True


def _family_classes(family):
    """``(config class, classifier class)`` of a family (transformers imported here)."""
    import transformers as t
    if family == "dinov2":
        return t.Dinov2Config, t.Dinov2ForImageClassification
    if family == "dinov2_with_registers":
        return t.Dinov2WithRegistersConfig, t.Dinov2WithRegistersForImageClassification
    if family == "convnextv2":
        return t.ConvNextV2Config, t.ConvNextV2ForImageClassification
    if family == "dinov3_vit":
        if not has_dinov3():
            raise RuntimeError("DINOv3 needs transformers >= 4.56 (the backbone study env; analysis/backbone_env.sh)")
        return _dinov3_classes()
    raise ValueError(f"unknown family {family!r}")


POS_EMBED_KEYS = ("pos_embed_shift", "pos_embed_jitter", "pos_embed_rescale")


def build_classifier(key, id2label, label2id, backbone_dir=None, pos_embed_aug="as_released", random_init=None,
                     arch=None):
    """A two-class classifier on backbone ``key`` with a newly initialised head, ``accepts_loss_kwargs = False``.

    ``pos_embed_aug`` (DINOv3 only): ``as_released`` keeps the config's train-time patch-coordinate augmentation
    (``pos_embed_shift/jitter/rescale``) as the hub config has it; ``off`` sets all three to None. ``random_init``
    (default: ``SVA_BACKBONE_RANDOM_INIT=1``) builds the config from the row's ``arch`` (or ``arch``, e.g. a tiny
    test size) and initialises every weight randomly, with no hub access: smoke tests only, recorded in provenance.

    Example::

        model = build_classifier("convnextv2-base", ID2LABEL, LABEL2ID)
    """
    if pos_embed_aug not in ("as_released", "off"):
        raise ValueError("pos_embed_aug must be 'as_released' or 'off'")
    b = get(key)
    cfg_cls, model_cls = _family_classes(b.family)
    random_init = is_random_init() if random_init is None else random_init
    head = {"num_labels": len(id2label), "id2label": dict(id2label), "label2id": dict(label2id)}
    pos = {k: None for k in POS_EMBED_KEYS} if (b.family == "dinov3_vit" and pos_embed_aug == "off") else {}
    if random_init:
        config = cfg_cls(**(b.arch if arch is None else arch), **head, **pos)
        model = model_cls(config)
    else:
        src, revision = resolve_source(b, backbone_dir)
        kwargs = {"revision": revision} if revision else {}
        if b.family != "dinov3_vit":
            kwargs["ignore_mismatched_sizes"] = True   # the hub head (ImageNet classes, or none) is replaced
        model = model_cls.from_pretrained(src, **kwargs, **head, **pos)
        if b.converted:
            # convert-rampnet saves its template with a two-class head, which from_pretrained would load unchanged,
            # so every seed would share it. Draw a new head here, under the caller's seed, as every other arm does.
            model._init_weights(model.classifier)
    model.accepts_loss_kwargs = False
    model._sva_random_init = bool(random_init)
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------------------------------

def file_sha256(path, chunk=1 << 24):
    """Hex sha256 of a file, read in 16 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _cached_file(repo_id, filename, revision):
    """Local path of ``filename`` of ``repo_id`` at ``revision`` in the HF cache, or None (no download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    p = try_to_load_from_cache(repo_id, filename, revision=revision)
    return p if isinstance(p, str) and os.path.isfile(p) else None


def _snapshot_sha(path):
    """The commit sha of an HF cache path ``.../snapshots/<sha>/<file>``, or None."""
    m = re.search(r"/snapshots/([0-9a-f]{40})/", path or "")
    return m.group(1) if m else None


def backbone_settings(key, model=None, backbone_dir=None):
    """The ``backbone`` block of ``training_run.json`` for a run on ``key`` with the built ``model``: registry facts,
    the revision the load resolved to, the parameter count of the built model, the preprocessing and its hash, the
    hash of the weights file actually loaded (None, with a warning, if it cannot be located; never a guess), the
    DINOv3 positional-augmentation values, and for a converted row the content of its ``conversion.json``.

    >>> s = backbone_settings("convnextv2-base")
    >>> s["key"], s["tokens_per_image"], s["pos_embed"], s["random_init"]
    ('convnextv2-base', 49, None, False)
    """
    b = get(key)
    random_init = bool(getattr(model, "_sva_random_init", False)) if model is not None else False
    out = {"key": b.key, "source": b.source, "revision": b.revision, "resolved_revision": None, "family": b.family,
           "model_class": type(model).__name__ if model is not None else MODEL_CLASS[b.family],
           "pooling": POOLING[b.family], "patch_size": b.patch_size, "num_prefix_tokens": b.num_prefix_tokens,
           "tokens_per_image": b.tokens_per_image,
           "params": int(sum(p.numel() for p in model.parameters())) if model is not None else None,
           "accepts_loss_kwargs": getattr(model, "accepts_loss_kwargs", None) if model is not None else None,
           "random_init": random_init, "pos_embed": None, "preprocess": preprocess_spec(b),
           "preprocess_sha256": preprocess_sha256(b), "weights_sha256": None, "conversion": None}
    if model is not None and b.family == "dinov3_vit":
        cfg = model.config
        out["pos_embed"] = {k.replace("pos_embed_", ""): getattr(cfg, k, None) for k in POS_EMBED_KEYS}
    if random_init:
        return out
    weights = None
    if b.converted:
        try:
            path, _ = resolve_source(b, backbone_dir)
        except FileNotFoundError:
            path = None
        if path:
            with open(os.path.join(path, CONVERSION_FILE)) as f:
                out["conversion"] = json.load(f)
            weights = os.path.join(path, "model.safetensors")
            weights = weights if os.path.isfile(weights) else None
    else:
        resolved = getattr(getattr(model, "config", None), "_commit_hash", None) if model is not None else None
        # The default key is loaded without a revision (as before #119), so it resolves through refs/main.
        weights = _cached_file(b.source, b.weights_file, resolved or (None if key == DEFAULT else b.revision))
        out["resolved_revision"] = resolved or _snapshot_sha(weights)
    if weights:
        out["weights_sha256"] = file_sha256(weights)
    elif model is not None:
        warnings.warn(f"{key}: could not locate the loaded weights file; weights_sha256 is null")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# RampNet -> ConvNextV2ForImageClassification
# ---------------------------------------------------------------------------------------------------------------------

_RAMPNET_RULES = [
    (re.compile(r"^0\.0\.(weight|bias)$"), r"convnextv2.embeddings.patch_embeddings.\1"),
    (re.compile(r"^0\.1\.(weight|bias)$"), r"convnextv2.embeddings.layernorm.\1"),
    (re.compile(r"^1\.(\d+)\.downsample\.([01])\.(weight|bias)$"),
     r"convnextv2.encoder.stages.\1.downsampling_layer.\2.\3"),
    (re.compile(r"^1\.(\d+)\.blocks\.(\d+)\.conv_dw\.(weight|bias)$"), r"convnextv2.encoder.stages.\1.layers.\2.dwconv.\3"),
    (re.compile(r"^1\.(\d+)\.blocks\.(\d+)\.norm\.(weight|bias)$"), r"convnextv2.encoder.stages.\1.layers.\2.layernorm.\3"),
    (re.compile(r"^1\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.(weight|bias)$"), r"convnextv2.encoder.stages.\1.layers.\2.pwconv1.\3"),
    (re.compile(r"^1\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.(weight|bias)$"), r"convnextv2.encoder.stages.\1.layers.\2.pwconv2.\3"),
    (re.compile(r"^1\.(\d+)\.blocks\.(\d+)\.mlp\.grn\.(weight|bias)$"), r"convnextv2.encoder.stages.\1.layers.\2.grn.\3"),
]


def rampnet_key_map(names):
    """``{rampnet name: transformers name}`` for RampNet's ``KeypointModel`` tensors (a timm ConvNeXt V2 as
    ``feature_extractor = Sequential(stem, stages)``, plus a heatmap ``head``). An optional leading ``model.`` is
    stripped, ``head.*`` is dropped, and any other name that no rule places raises ``ValueError``.

    >>> rampnet_key_map(["model.feature_extractor.0.0.weight", "model.head.0.bias",
    ...                  "feature_extractor.1.2.blocks.5.mlp.grn.bias"])
    {'model.feature_extractor.0.0.weight': 'convnextv2.embeddings.patch_embeddings.weight', 'feature_extractor.1.2.blocks.5.mlp.grn.bias': 'convnextv2.encoder.stages.2.layers.5.grn.bias'}
    """
    out = {}
    for name in names:
        n = name[len("model."):] if name.startswith("model.") else name
        if n.startswith("head."):
            continue
        if not n.startswith("feature_extractor."):
            raise ValueError(f"cannot place RampNet tensor {name!r}")
        rest = n[len("feature_extractor."):]
        for pattern, repl in _RAMPNET_RULES:
            if pattern.match(rest):
                out[name] = pattern.sub(repl, rest)
                break
        else:
            raise ValueError(f"cannot place RampNet tensor {name!r}")
    return out


def is_grn(hf_name):
    """True for a GRN tensor, whose shape is ``(4C,)`` in timm and ``(1, 1, 1, 4C)`` in transformers."""
    return ".grn." in hf_name


def plan_conversion(src_shapes, template_shapes):
    """Check that ``rampnet_key_map`` of ``src_shapes`` (``{name: shape}``) is a bijection onto every
    ``convnextv2.embeddings.*`` and ``convnextv2.encoder.*`` tensor of ``template_shapes``, with equal shapes after
    the GRN reshape. Returns ``{src name: hf name}``; ``ValueError`` listing what is missing, extra or mis-shaped.
    The template's final ``convnextv2.layernorm`` and the classifier are not mapped (they keep the template's
    values / are new)."""
    mapping = rampnet_key_map(src_shapes)
    targets = {k for k in template_shapes if k.startswith(("convnextv2.embeddings.", "convnextv2.encoder."))}
    mapped = list(mapping.values())
    dupes = sorted({v for v in mapped if mapped.count(v) > 1})
    missing = sorted(targets - set(mapped))
    extra = sorted(set(mapped) - targets)
    bad = []
    for s, t in mapping.items():
        if t not in template_shapes:
            continue
        shape = list(src_shapes[s])
        if is_grn(t):
            shape = [1, 1, 1] + shape
        if shape != list(template_shapes[t]):
            bad.append(f"{s} {list(src_shapes[s])} -> {t} {list(template_shapes[t])}")
    if dupes or missing or extra or bad:
        raise ValueError(f"RampNet -> ConvNeXt V2 map is not a bijection with equal shapes: duplicates {dupes[:5]}, "
                         f"template tensors not covered {missing[:5]} ({len(missing)}), unknown targets {extra[:5]}, "
                         f"shape mismatches {bad[:5]}")
    return mapping


def convert_state_dict(src, template):
    """``template`` (a ``ConvNextV2ForImageClassification`` state dict) with every embeddings/encoder tensor
    replaced by RampNet's (``src``, a state dict with ``model.feature_extractor.*`` names), after
    ``plan_conversion``. Returns ``(state_dict, counts)`` where counts gives the tensors by origin."""
    mapping = plan_conversion({k: list(v.shape) for k, v in src.items()}, {k: list(v.shape) for k, v in template.items()})
    out = dict(template)
    for s, t in mapping.items():
        out[t] = src[s].reshape(template[t].shape).to(template[t].dtype).clone()
    counts = {"from_rampnet": len(mapping), "dropped_rampnet_head": len(src) - len(mapping),
              "kept_from_template": sum(1 for k in template if k.startswith("convnextv2.") and k not in mapping.values()),
              "new_classifier": sum(1 for k in template if k.startswith("classifier."))}
    return out, counts


def timm_feature_extractor(src, timm_name="convnextv2_base", **timm_kwargs):
    """timm's ``timm_name`` (random init; ``timm_kwargs`` to ``timm.create_model``) as RampNet's
    ``feature_extractor`` (``Sequential(stem, stages)``), with ``src``'s ``model.feature_extractor.*`` tensors loaded
    strictly, in eval mode."""
    import timm
    from torch import nn
    net = timm.create_model(timm_name, pretrained=False, **timm_kwargs)
    fe = nn.Sequential(*list(net.children())[:-2])
    sd = {}
    for k, v in src.items():
        n = k[len("model."):] if k.startswith("model.") else k
        if n.startswith("feature_extractor."):
            sd[n[len("feature_extractor."):]] = v
    fe.load_state_dict(sd, strict=True)
    return fe.eval()


def timm_parity(src, hf_model, timm_name="convnextv2_base", size=224, seed=0, **timm_kwargs):
    """Max absolute difference between timm's feature map (``timm_feature_extractor``) and the converted model's
    last stage output (``hidden_states[-1]``) on a seeded ``1 x 3 x size x size`` input."""
    import torch
    fe = timm_feature_extractor(src, timm_name, **timm_kwargs)
    x = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(seed))
    hf_model = hf_model.eval()
    with torch.no_grad():
        a = fe(x)
        b = hf_model.convnextv2(x, output_hidden_states=True).hidden_states[-1]
    return float((a - b).abs().max())


def convert_rampnet(out, key="rampnet-convnextv2-base", verify_with_timm=False, tol=1e-4):
    """Convert RampNet's ConvNeXt V2-B weights (row ``key``) into a native ``ConvNextV2ForImageClassification`` at
    ``out``: template ``convnextv2-base`` (two classes, new head), stem and stages replaced through
    ``plan_conversion``; the final layernorm keeps the template's values, so this arm and the ``convnextv2-base``
    control differ only in stem and stage weights. Writes ``config.json``, ``model.safetensors`` and
    ``conversion.json``. With ``verify_with_timm``, refuses unless ``timm_parity`` < ``tol``. Skips (returns the
    record) when ``conversion.json`` already says verified."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from fine_tune_classifier import ID2LABEL, LABEL2ID, git_state
    b, tb = get(key), get("convnextv2-base")
    record_path = os.path.join(out, CONVERSION_FILE)
    if os.path.isfile(record_path):
        with open(record_path) as f:
            rec = json.load(f)
        if rec.get("verified_against_timm") or not verify_with_timm:
            print(f"{out} already converted (verified_against_timm={rec.get('verified_against_timm')})")
            return rec
    weights = hf_hub_download(b.source, b.weights_file, revision=b.revision)
    src = load_file(weights)
    _, cls = _family_classes("convnextv2")
    template = cls.from_pretrained(tb.source, revision=tb.revision, num_labels=len(ID2LABEL), id2label=ID2LABEL,
                                   label2id=LABEL2ID, ignore_mismatched_sizes=True)
    sd, counts = convert_state_dict(src, template.state_dict())
    template.load_state_dict(sd, strict=True)
    rec = {"key": key, "weights_repo": b.source, "weights_revision": b.revision, "weights_file": b.weights_file,
           "weights_sha256": file_sha256(weights), "template": tb.source, "template_revision": tb.revision,
           "tensor_counts": counts,
           "note": "stem and stages from RampNet; the final convnextv2.layernorm keeps the template's ImageNet values "
                   "(as the convnextv2-base control has them); the saved classifier is a placeholder: build_classifier "
                   "draws a new one under each run's seed",
           "versions": {m: getattr(__import__(m), "__version__", None) for m in ("torch", "transformers")},
           **git_state(), "verified_against_timm": False, "timm_max_abs_diff": None}
    if verify_with_timm:
        import timm
        diff = timm_parity(src, template)
        rec.update(timm_max_abs_diff=diff, timm_version=timm.__version__)
        print(f"timm parity: max |timm - hf| = {diff:.3g} (tolerance {tol})")
        if not diff < tol:
            raise SystemExit(f"RampNet conversion does not match timm: max abs diff {diff} >= {tol}")
        rec["verified_against_timm"] = True
    os.makedirs(out, exist_ok=True)
    template.save_pretrained(out)
    with open(record_path, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    print(f"wrote {out} ({counts})")
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# prefetch (login node) and storage check
# ---------------------------------------------------------------------------------------------------------------------

def _default_download(repo_id, revision, allow_patterns):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, revision=revision, allow_patterns=allow_patterns)


def _failure_reason(exc, token=None):
    """A short reason for a failed download, from the exception's class name and HTTP status (no
    ``huggingface_hub.errors`` import, so CI without huggingface_hub can test it); a token never appears."""
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if name == "GatedRepoError" or status == 403:
        reason = GATED_REASON
    else:
        reason = f"{name}" + (f" (HTTP {status})" if status else "") + f": {str(exc).splitlines()[0][:200] if str(exc) else ''}"
    return reason.replace(token, "<token>") if token else reason


def _refs_main(snapshot_dir):
    """The sha ``refs/main`` points to in the HF cache repo that ``snapshot_dir`` belongs to, or None."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(snapshot_dir)))
        with open(os.path.join(repo, "refs", "main")) as f:
            return f.read().strip()
    except OSError:
        return None


def prefetch(keys, download=None, env=None, hash_files=True):
    """Download each row of ``keys`` (config, preprocessor config and its weights file only) at its pinned revision.
    A gated row is not attempted unless ``HF_TOKEN`` is set. Returns the manifest ``{key: {...}}``: source,
    revision, ``available``, ``reason`` when unavailable, and the sha256 and size of each file; for the default key
    also ``refs_main`` (the revision an unpinned load resolves to), which must equal the pin. ``download(repo_id,
    revision, allow_patterns)`` returns the snapshot directory (default ``huggingface_hub.snapshot_download``)."""
    env = os.environ if env is None else env
    download = download or _default_download
    token = env.get("HF_TOKEN") or None
    manifest = {}
    for key in keys:
        b = get(key)
        rec = {"source": b.source, "revision": b.revision, "gated": b.gated, "available": False}
        if b.gated and not token:
            rec["reason"] = GATED_REASON
            manifest[key] = rec
            continue
        patterns = ["config.json", "preprocessor_config.json", b.weights_file]
        try:
            snap = download(b.source, b.revision, patterns)
        except Exception as e:  # noqa: BLE001 -- every failure is recorded per row; the others still run
            rec["reason"] = _failure_reason(e, token)
            manifest[key] = rec
            continue
        files = {}
        for name in patterns:
            p = os.path.join(snap, name)
            if os.path.isfile(p):
                files[name] = {"size": os.path.getsize(p), "sha256": file_sha256(p) if hash_files else None}
        if b.weights_file not in files:
            rec["reason"] = f"{b.weights_file} not in the snapshot"
        else:
            rec.update(available=True, files=files, snapshot_revision=_snapshot_sha(os.path.join(snap, "x")) or b.revision)
        if key == DEFAULT and rec["available"]:
            rec["refs_main"] = _refs_main(snap)
            if rec["refs_main"] != b.revision:
                rec.update(available=False, reason=f"refs/main is {rec['refs_main']}, not the pin {b.revision}: "
                                                   "an unpinned load would read other weights")
        manifest[key] = rec
    return manifest


def parse_hyakstorage(text):
    """``(free_gb, free_files)`` from the ``Total:`` row of ``hyakstorage <fileset>`` output, or None.

    >>> parse_hyakstorage("│ Total:  │ 969GB / 1024GB  │ 974033 / 1000000 files │")
    (55, 25967)
    """
    for line in text.splitlines():
        if "Total:" in line:
            m = re.search(r"([\d.]+)\s*([GT])B\s*/\s*([\d.]+)\s*([GT])B.*?(\d+)\s*/\s*(\d+)\s*files", line)
            if m:
                scale = {"G": 1, "T": 1024}
                used, cap = float(m[1]) * scale[m[2]], float(m[3]) * scale[m[4]]
                return int(cap - used), int(m[6]) - int(m[5])
    return None


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def registry_markdown():
    """The registry as a markdown table (pasted into docs/backbone-2026-11.md)."""
    lines = ["| key | source | revision | family | patch | prefix tokens | tokens / image | params (M) | input | gated |"
             " converted | eligible |", "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|"]
    for b in REGISTRY.values():
        lines.append(f"| `{b.key}` | `{b.source}` | `{b.revision[:12]}` | {b.family} | {b.patch_size or '32 (stride)'} "
                     f"| {b.num_prefix_tokens} | {b.tokens_per_image} | {b.params_m} | {b.resize_px}/{b.crop_px} | "
                     f"{'yes' if b.gated else 'no'} | {'yes' if b.converted else 'no'} | "
                     f"{'yes' if b.key in ELIGIBLE else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the registry as a markdown table")
    p = sub.add_parser("prefetch", help="download the non-gated rows' weights at their pinned revisions")
    p.add_argument("--backbones", required=True, help="comma-separated registry keys")
    p.add_argument("--out", required=True, help="manifest JSON to write")
    c = sub.add_parser("convert-rampnet", help="convert RampNet's backbone into ConvNextV2ForImageClassification")
    c.add_argument("--out", required=True)
    c.add_argument("--key", default="rampnet-convnextv2-base", choices=[k for k, b in REGISTRY.items() if b.converted])
    c.add_argument("--verify-with-timm", action="store_true")
    s = sub.add_parser("storage-free", help="read hyakstorage output on stdin; exit 4 below the minimums")
    s.add_argument("--min-gb", type=float, required=True)
    s.add_argument("--min-files", type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "list":
        print(registry_markdown(), end="")
        return 0
    if args.cmd == "prefetch":
        keys = [k.strip() for k in args.backbones.split(",") if k.strip()]
        for k in keys:
            get(k)
        manifest = prefetch(keys)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"backbones": manifest, "hf_home": os.environ.get("HF_HOME")}, f, indent=2)
            f.write("\n")
        for k, rec in manifest.items():
            print(f"{k}: {'available' if rec['available'] else 'UNAVAILABLE: ' + rec.get('reason', '')}")
        return 0 if all(r["available"] for r in manifest.values()) else 3
    if args.cmd == "convert-rampnet":
        convert_rampnet(args.out, args.key, args.verify_with_timm)
        return 0
    if args.cmd == "storage-free":
        free = parse_hyakstorage(sys.stdin.read())
        if free is None:
            print("could not read the Total row of hyakstorage", file=sys.stderr)
            return 4
        gb, files = free
        print(f"free: {gb} GB, {files} files (need {args.min_gb} GB, {args.min_files} files)")
        return 0 if gb >= args.min_gb and files >= args.min_files else 4
    return 2


__all__ = ["Backbone", "DEFAULT", "REGISTRY", "ELIGIBLE", "get", "preprocess_spec", "build_image_processor",
           "resolve_source", "build_classifier", "register_auto", "backbone_settings", "rampnet_key_map",
           "plan_conversion", "convert_rampnet", "prefetch", "parse_hyakstorage"]

if __name__ == "__main__":
    sys.exit(main())
