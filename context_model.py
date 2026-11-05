"""The "more context" validator arms of issue #14: three ways to show the DINOv2 classifier more than the tight crop.

=================  ==========================================================================  =====================
arm                input                                                                       head input
=================  ==========================================================================  =====================
``two_scale``      the tight crop and the wide crop (``context_crops.py``, 4x the half-width),  ``4 x hidden``
                   one shared backbone, ``[CLS, mean(patches)]`` of each, concatenated
``view_heatmap``   the whole label-centred view (448 px stored, ``--context-view-res`` 224 by   ``2 x hidden``
                   default) plus a 4th channel: a Gaussian at the label point whose width is
                   the tight crop's extent
``side_info``      the tight crop plus a normalised vector of non-image features from          ``2 x hidden + n``
                   split.csv (severity, tag count, imagery age, log depth, high-quality user,
                   clipped, and missing-value flags)
=================  ==========================================================================  =====================

Every arm pools like ``Dinov2ForImageClassification`` (``[CLS, mean of the patch tokens]``) and trains with the same
recipe as round 3 through ``fine_tune_classifier.py --context-arm <arm>``.

**Why the heatmap encodes scale.** At evaluation the label is always at the view centre, so a fixed-width Gaussian
would be the same image for every label: a constant channel the model can only ignore. Its width is the tight crop's
half-width mapped into the input (``heatmap_sigma``: half of the crop's half-width, at least 4 px), so the channel
says both *where* the label is and *how big* the depth-sized crop around it is. The issue's wording was "a Gaussian
at the label position"; the plain point is ``heatmap_sigma(..., floor=s)`` with ``crop_half_px=0``.

**Why the view skips the centre crop.** The checkpoint's processor resizes to 256 and centre-crops 224, which drops the
outer 12.5% of each side: for a whole-view input that is exactly the context the arm is about. The view arm resizes to
``res`` x ``res`` instead (train: ``RandomResizedCrop`` with the heatmap mapped through the same crop and flip).

**The 4-channel patch embedding** copies the pretrained RGB filters into channels 0-2 and zero-initialises channel 3,
so the model computes exactly the pretrained function at step 0 and learns to use the heatmap.

torch, torchvision and transformers are imported inside functions and the classes are built by ``build_classes()``, so
this module imports on the CPU-only CI. The model is never registered with ``AutoModel`` (``model_type``
``context_dinov2``): ``evaluate.py`` and sidewalk-ai-api cannot load it by accident (``publish_to_hf.py`` copies
checkpoint files without loading them, and ``--context-arm`` never writes to ``<Type>/dinov2/``). Score a checkpoint
with this module, then let ``evaluate.py`` compute the metrics::

    python context_model.py predict --label_type NoCurbRamp --data-dir D --checkpoint RUN/checkpoint-N \
        --tag view_heatmap
    python evaluate.py --label_type NoCurbRamp --data-dir D --skip-inference --tag view_heatmap --per-city

Pattern: ``region_model.py`` (issue #33, PR #88).
"""
import argparse
import functools
import os
import sys

import numpy as np
import pandas as pd

ARMS = ("two_scale", "view_heatmap", "side_info")
MODEL_TYPE = "context_dinov2"
SIDE_FEATURES = ["severity", "n_tags", "age_years", "log_depth_m", "high_quality_user", "clipped"]
SIDE_MISSING = ["severity_missing", "age_missing", "depth_missing"]
SIDE_COLUMNS = SIDE_FEATURES + SIDE_MISSING
#: split.csv columns each arm reads (besides image and label), for ``fine_tune_classifier.extra_frame_columns``.
FRAME_COLUMNS = {
    "two_scale": ("label_uid", "split", "crop_half_px"),
    "view_heatmap": ("label_uid", "split", "crop_half_px"),
    "side_info": ("label_uid", "split", "severity", "tags", "image_capture_date", "depth_m", "high_quality_user",
                  "clipped"),
}
VIEW_SIZE = 2048            # data_generation.VIEW_SIZE (not imported: that module pulls in the generation stack)
HEATMAP_SIGMA_FLOOR = 4.0   # px at the model input
TRAIN_SCALE = (0.7, 1.0)    # fine_tune_classifier.py's RandomResizedCrop scale
TRAIN_RATIO = (3 / 4, 4 / 3)


# ---------------------------------------------------------------------------------------------------------------------
# Side information (pure pandas)
# ---------------------------------------------------------------------------------------------------------------------

def _capture_month(values):
    """``image_capture_date`` (``YYYY-MM`` or a longer ISO date) as a Timestamp of the month; NaT when unparseable."""
    return pd.to_datetime(pd.Series(values).astype(str).str.slice(0, 7), format="%Y-%m", errors="coerce")


def age_reference(df):
    """The reference date for ``age_years``: the latest ``image_capture_date`` in ``df`` (``YYYY-MM``), or None.

    >>> age_reference(pd.DataFrame({"image_capture_date": ["2019-04", None, "2024-09"]}))
    '2024-09'
    """
    d = _capture_month(df["image_capture_date"]).max()
    return None if pd.isna(d) else d.strftime("%Y-%m")


def _flag(v):
    return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0


def side_feature_frame(df, reference_date=None):
    """The raw side features of each row of ``df`` (split.csv columns), as ``SIDE_COLUMNS``.

    ``severity`` as a float; ``n_tags`` the number of tags (``analysis.slices.parse_tags``: JSON list or ``;``-joined);
    ``age_years`` = (``reference_date`` - ``image_capture_date``) in years (``reference_date`` defaults to
    ``age_reference(df)``); ``log_depth_m`` = log of a positive ``depth_m``; ``high_quality_user`` and ``clipped`` as
    0/1 (missing -> 0); and a 0/1 ``*_missing`` flag for severity, age and depth, whose own values stay NaN here
    (``normalise`` fills them with the training median).

    >>> f = side_feature_frame(pd.DataFrame({"severity": [3, None], "tags": ['["a","b"]', None],
    ...     "image_capture_date": ["2020-09", "2024-09"], "depth_m": [10.0, 0], "high_quality_user": [True, None],
    ...     "clipped": [0, 1]}))
    >>> f[["severity", "n_tags", "age_years", "high_quality_user", "clipped"]].values.tolist()
    [[3.0, 2.0, 4.0, 1.0, 0.0], [nan, 0.0, 0.0, 0.0, 1.0]]
    >>> f[SIDE_MISSING].values.tolist()
    [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    """
    from analysis.slices import parse_tags
    ref = pd.Timestamp(reference_date or age_reference(df) or "2000-01")
    out = pd.DataFrame(index=df.index)
    out["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype(float)
    out["n_tags"] = [float(len(parse_tags(t))) for t in df["tags"]]
    cap = _capture_month(df["image_capture_date"]).set_axis(df.index)
    out["age_years"] = ((ref - cap).dt.days / 365.25).round(6)
    depth = pd.to_numeric(df["depth_m"], errors="coerce").astype(float)
    out["log_depth_m"] = np.log(depth.where(depth > 0))
    out["high_quality_user"] = [_flag(v) for v in df["high_quality_user"]]
    out["clipped"] = [_flag(v) for v in df["clipped"]]
    out["severity_missing"] = out["severity"].isna().astype(float)
    out["age_missing"] = out["age_years"].isna().astype(float)
    out["depth_missing"] = out["log_depth_m"].isna().astype(float)
    return out[SIDE_COLUMNS]


def fit_normaliser(feats):
    """Per-column ``median`` (NaN fill), ``mean`` and ``std`` (floored at 1e-6) of the training rows' side features,
    as JSON-ready lists in ``SIDE_COLUMNS`` order. A column with no value at all gets median 0.

    >>> s = fit_normaliser(pd.DataFrame({c: [1.0, 3.0] for c in SIDE_COLUMNS}))
    >>> s["median"][0], s["mean"][0], s["std"][0]
    (2.0, 2.0, 1.0)
    """
    x = feats[SIDE_COLUMNS].to_numpy(dtype=float)
    med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in x.T])
    filled = np.where(np.isnan(x), med, x)
    return {"columns": list(SIDE_COLUMNS), "median": [float(v) for v in med],
            "mean": [float(v) for v in filled.mean(axis=0)],
            "std": [float(max(v, 1e-6)) for v in filled.std(axis=0)]}


def normalise(feats, stats):
    """``float32`` array: NaN -> the training median, then ``(x - mean) / std``."""
    if list(stats["columns"]) != list(SIDE_COLUMNS):
        raise ValueError(f"side-feature columns differ: {stats['columns']} vs {SIDE_COLUMNS}")
    x = feats[SIDE_COLUMNS].to_numpy(dtype=float)
    x = np.where(np.isnan(x), np.asarray(stats["median"]), x)
    return ((x - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# The point heatmap (numpy)
# ---------------------------------------------------------------------------------------------------------------------

def heatmap(h, w, cx, cy, sigma):
    """An ``h`` x ``w`` float32 Gaussian with peak 1.0 at pixel index ``(cx, cy)``; ``sigma`` in px, a number or
    ``(sigma_x, sigma_y)`` (a non-square training crop stretches it).

    >>> m = heatmap(9, 9, 4, 4, 2.0)
    >>> float(m[4, 4]), round(float(m[4, 6]), 4), bool(m[4, 2] == m[4, 6] == m[2, 4])
    (1.0, 0.6065, True)
    """
    sx, sy = (sigma, sigma) if np.isscalar(sigma) else sigma
    x = (np.arange(w, dtype=np.float32) - cx) / sx
    y = (np.arange(h, dtype=np.float32) - cy) / sy
    return np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2)).astype(np.float32)


def heatmap_sigma(crop_half_px, res, view_size=VIEW_SIZE, floor=HEATMAP_SIGMA_FLOOR):
    """The heatmap's sigma in model-input px for a view resized to ``res``: half of the tight crop's half-width
    (``crop_half_px`` view px), at least ``floor``.

    >>> heatmap_sigma(238, 224), heatmap_sigma(20, 224), heatmap_sigma(238, 448)
    (13.015625, 4.0, 26.03125)
    """
    return max(floor, float(crop_half_px) * res / view_size / 2)


def map_point(cx, cy, crop, res, flip=False):
    """Where pixel index ``(cx, cy)`` of an image lands after ``torchvision``'s ``resized_crop(i, j, h, w) -> res``
    and an optional horizontal flip, in pixel index coordinates (pixel centres are mapped, hence the half pixels).

    >>> map_point(10, 10, (0, 0, 20, 20), 20), map_point(10, 10, (0, 0, 20, 20), 20, flip=True)
    ((10.0, 10.0), (9.0, 10.0))
    >>> map_point(13.5, 9.5, (4, 8, 8, 8), 16)
    (11.5, 11.5)
    """
    i, j, h, w = crop
    x = (cx + 0.5 - j) * res / w - 0.5
    y = (cy + 0.5 - i) * res / h - 0.5
    return (res - 1 - x if flip else x), y


# ---------------------------------------------------------------------------------------------------------------------
# Input transforms (torch / torchvision, imported when called)
# ---------------------------------------------------------------------------------------------------------------------

class ContextInputs:
    """Turns one example's context into tensors for the model, train (augmented) or eval (deterministic).

    ``paths`` maps label_uid -> (wide_path, view_path); ``tight`` maps label_uid -> the tight crop's path (train
    rows of ``two_scale``); ``side`` maps label_uid -> normalised side vector. In training, ``two_scale`` re-draws
    the tight crop's augmentation together with the wide crop's (:meth:`pair`): each gets its own
    ``RandomResizedCrop(crop, scale=(0.7, 1.0))`` but they share ONE flip, so the two streams are never mirrored
    against each other (they never are at evaluation). At evaluation both go through the checkpoint's own processor
    (resize, centre crop); the view as described in the module docstring.
    """

    def __init__(self, arm, image_processor, res=224, paths=None, side=None, sigma_floor=HEATMAP_SIGMA_FLOOR,
                 tight=None):
        self.arm, self.processor, self.res = arm, image_processor, int(res)
        self.paths, self.side, self.sigma_floor = paths or {}, side or {}, sigma_floor
        self.tight = tight or {}
        crop = image_processor.crop_size
        self.crop = (crop["height"], crop["width"]) if "height" in crop else crop["shortest_edge"]
        self.mean, self.std = list(image_processor.image_mean), list(image_processor.image_std)

    def _open(self, uid, which):
        from PIL import Image
        return Image.open(self.paths[uid][0 if which == "wide" else 1]).convert("RGB")

    def wide(self, uid, train=False):
        """The wide crop at evaluation (the processor's resize and centre crop); training goes through :meth:`pair`."""
        if train:
            return self.pair(uid)[1]
        return self.processor(self._open(uid, "wide"), return_tensors="pt")["pixel_values"][0]

    def pair(self, uid):
        """``(tight, wide)`` training tensors: independent scale jitter, one shared horizontal flip."""
        import torch
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
        flip = bool(torch.rand(1) < 0.5)
        out = []
        for img in (Image.open(self.tight[uid]).convert("RGB"), self._open(uid, "wide")):
            tf = Compose([RandomResizedCrop(self.crop, scale=TRAIN_SCALE, antialias=True), ToTensor(),
                          Normalize(mean=self.mean, std=self.std)])
            t = tf(img)
            out.append(t.flip(-1) if flip else t)
        return tuple(out)

    def view(self, uid, crop_half_px, train):
        """The 4-channel view: normalised RGB at ``res`` plus the heatmap."""
        import torch
        import torchvision.transforms.functional as F
        from torchvision.transforms import RandomResizedCrop
        img = self._open(uid, "view")
        s = img.size[0]
        centre = (s - 1) / 2
        if train:
            i, j, h, w = RandomResizedCrop.get_params(img, scale=TRAIN_SCALE, ratio=TRAIN_RATIO)
            flip = bool(torch.rand(1) < 0.5)
            img = F.resized_crop(img, i, j, h, w, [self.res, self.res], antialias=True)
            if flip:
                img = F.hflip(img)
            cx, cy = map_point(centre, centre, (i, j, h, w), self.res, flip)
            raw = float(crop_half_px) * s / VIEW_SIZE / 2  # sigma in stored-view px, then through the crop
            sigma = (max(self.sigma_floor, raw * self.res / w), max(self.sigma_floor, raw * self.res / h))
        else:
            img = F.resize(img, [self.res, self.res], antialias=True)
            cx, cy = map_point(centre, centre, (0, 0, s, s), self.res)
            sigma = heatmap_sigma(crop_half_px, self.res, floor=self.sigma_floor)
        rgb = F.normalize(F.to_tensor(img), self.mean, self.std)
        hm = torch.from_numpy(heatmap(self.res, self.res, cx, cy, sigma))[None]
        return torch.cat([rgb, hm], dim=0)

    def side_vector(self, uid):
        import torch
        return torch.as_tensor(self.side[uid], dtype=torch.float32)


def wrap_collate(collate_fn, inputs):
    """A collate that adds the arm's inputs to ``collate_fn``'s batch (hard or soft-target alike): reads each
    example's ``label_uid`` and ``split`` (train rows augmented, the rest not) and, for the view, ``crop_half_px``.
    ``two_scale`` adds ``pixel_values_wide`` and, for train rows, replaces ``pixel_values`` with the tight crop
    re-augmented under the same flip (:meth:`ContextInputs.pair`); ``view_heatmap`` replaces ``pixel_values`` with
    the 4-channel view; ``side_info`` adds ``side_features``."""
    def collate(examples):
        import torch
        batch = collate_fn(examples)
        train = [str(e["split"]) == "train" for e in examples]
        uids = [str(e["label_uid"]) for e in examples]
        if inputs.arm == "two_scale":
            tight, wide = [], []
            for i, (u, t) in enumerate(zip(uids, train)):
                a, b = inputs.pair(u) if t else (batch["pixel_values"][i], inputs.wide(u))
                tight.append(a)
                wide.append(b)
            batch["pixel_values"], batch["pixel_values_wide"] = torch.stack(tight), torch.stack(wide)
        elif inputs.arm == "view_heatmap":
            batch["pixel_values"] = torch.stack([inputs.view(u, e["crop_half_px"], t)
                                                 for u, e, t in zip(uids, examples, train)])
        elif inputs.arm == "side_info":
            batch["side_features"] = torch.stack([inputs.side_vector(u) for u in uids])
        return batch
    return collate


# ---------------------------------------------------------------------------------------------------------------------
# The model (torch + transformers)
# ---------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def build_classes():
    """``(ContextDinov2Config, ContextDinov2ForImageClassification)``, built once (transformers imported here)."""
    import torch
    from torch import nn
    from transformers import Dinov2Config, Dinov2Model, Dinov2PreTrainedModel
    from transformers.modeling_outputs import ImageClassifierOutput

    class ContextDinov2Config(Dinov2Config):
        model_type = MODEL_TYPE

        def __init__(self, context_arm="two_scale", view_px=448, train_res=224, wide_factor=4,
                     heatmap_sigma_floor=HEATMAP_SIGMA_FLOOR, side_features=None, side_stats=None,
                     age_reference_date=None, index_sha256=None, **kwargs):
            super().__init__(**kwargs)
            if context_arm not in ARMS:
                raise ValueError(f"context_arm must be one of {ARMS}")
            self.context_arm = context_arm
            self.view_px, self.train_res, self.wide_factor = int(view_px), int(train_res), int(wide_factor)
            self.heatmap_sigma_floor = float(heatmap_sigma_floor)
            self.side_features = list(side_features) if side_features else []
            self.side_stats = side_stats
            self.age_reference_date = age_reference_date
            self.index_sha256 = index_sha256

    class ContextDinov2ForImageClassification(Dinov2PreTrainedModel):
        config_class = ContextDinov2Config
        accepts_loss_kwargs = False  # as region_model: the Trainer divides by the accumulation steps itself

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.dinov2 = Dinov2Model(config)
            h = config.hidden_size
            n_in = {"two_scale": 4 * h, "view_heatmap": 2 * h,
                    "side_info": 2 * h + len(config.side_features)}[config.context_arm]
            self.classifier = nn.Linear(n_in, config.num_labels)
            self.post_init()

        def pool(self, pixel_values):
            seq = self.dinov2(pixel_values=pixel_values).last_hidden_state
            return torch.cat([seq[:, 0], seq[:, 1:].mean(dim=1)], dim=1)  # Dinov2ForImageClassification's pooling

        def forward(self, pixel_values=None, pixel_values_wide=None, side_features=None, labels=None):
            feats = [self.pool(pixel_values)]
            arm = self.config.context_arm
            if arm == "two_scale":
                if pixel_values_wide is None:
                    raise ValueError("the two_scale arm needs pixel_values_wide")
                feats.append(self.pool(pixel_values_wide))
            elif arm == "side_info":
                if side_features is None:
                    raise ValueError("the side_info arm needs side_features")
                feats.append(side_features.to(feats[0].dtype))
            logits = self.classifier(torch.cat(feats, dim=1))
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))
            return ImageClassifierOutput(loss=loss, logits=logits)

        @classmethod
        def from_base(cls, base, context_arm, id2label=None, label2id=None, **fields):
            """A new model whose backbone is ``base``'s (``facebook/dinov2-large`` or a checkpoint directory) and
            whose head is fresh. For ``view_heatmap`` the patch projection gets a 4th input channel: channels 0-2
            are the pretrained filters, channel 3 is zero, so the output equals the base model's at step 0."""
            base_cfg = Dinov2Config.from_pretrained(base).to_dict()
            for k in ("model_type", "architectures", "transformers_version", "id2label", "label2id", "num_labels",
                      "num_channels", "context_arm", "view_px", "train_res", "wide_factor", "heatmap_sigma_floor",
                      "side_features", "side_stats", "age_reference_date", "index_sha256"):
                base_cfg.pop(k, None)
            channels = 4 if context_arm == "view_heatmap" else 3
            cfg = ContextDinov2Config(context_arm=context_arm, id2label=id2label, label2id=label2id,
                                      num_channels=channels, **fields, **base_cfg)
            model = cls(cfg)
            state = Dinov2Model.from_pretrained(base).state_dict()
            key = "embeddings.patch_embeddings.projection.weight"
            if channels == 4:
                w = state[key]
                state[key] = torch.cat([w, torch.zeros_like(w[:, :1])], dim=1)
            model.dinov2.load_state_dict(state)
            return model

    return ContextDinov2Config, ContextDinov2ForImageClassification


def side_table(df, reference_date, stats):
    """``{label_uid: normalised side vector}`` for the rows of ``df``."""
    vecs = normalise(side_feature_frame(df, reference_date), stats)
    return dict(zip(df["label_uid"].astype(str), vecs))


def build_for_training(args, base, train_df, val_df, id2label, label2id, collate_fn, image_processor):
    """The arm's model and collate for ``fine_tune_classifier.py --context-arm``: ``(model, collate_fn)``.

    Needs split.csv frames carrying ``FRAME_COLUMNS[arm]`` (``load_split_frames(..., extra_columns=...)``). For
    ``two_scale``/``view_heatmap`` every train and val row must be in ``<data-dir>/<label_type>/context/index.csv``
    (``context_crops.load_index`` raises otherwise, before any weight is loaded) and the index's sha256 goes into the
    config; for ``side_info`` the normaliser is fitted on the train rows and saved in the config with the age
    reference date, so inference applies the same transform.
    """
    import context_crops
    arm = args.context_arm
    need = FRAME_COLUMNS[arm]
    if train_df is None or any(c not in train_df.columns for c in need):
        raise SystemExit(f"--context-arm {arm} needs split.csv frames with {list(need)} (not the imagefolder layout)")
    res = int(getattr(args, "context_view_res", 224))
    fields = {"train_res": res, "wide_factor": context_crops.WIDE_FACTOR, "view_px": context_crops.VIEW_PX}
    paths, side, tight = None, None, None
    if arm in ("two_scale", "view_heatmap"):
        base_dir = os.path.join(args.data_dir, args.label_type)
        both = pd.concat([train_df, val_df], ignore_index=True)
        index = context_crops.load_index(base_dir, require_uids=both["label_uid"],
                                         crop_half_px=dict(zip(both["label_uid"].astype(str), both["crop_half_px"])))
        paths = dict(zip(index["label_uid"], zip(index["wide_path"], index["view_path"])))
        fields["index_sha256"] = context_crops.index_sha256(base_dir)
        fields["wide_factor"], fields["view_px"] = context_crops.context_settings(base_dir)
        if arm == "two_scale":
            if "image" not in train_df.columns:
                raise SystemExit("--context-arm two_scale needs the train frame's image paths")
            tight = dict(zip(train_df["label_uid"].astype(str), train_df["image"]))
    else:
        ref = age_reference(train_df)
        stats = fit_normaliser(side_feature_frame(train_df, ref))
        side = {**side_table(train_df, ref, stats), **side_table(val_df, ref, stats)}
        fields.update(side_features=list(SIDE_COLUMNS), side_stats=stats, age_reference_date=ref)
    print(f"context arm {arm}: {len(train_df)} train / {len(val_df)} val rows"
          + (f", view at {res} px" if arm == "view_heatmap" else ""))
    _, model_cls = build_classes()
    model = model_cls.from_base(base, arm, id2label, label2id, **fields)
    inputs = ContextInputs(arm, image_processor, res, paths, side, model.config.heatmap_sigma_floor, tight)
    return model, wrap_collate(collate_fn, inputs)


def run_inference(manifest, checkpoint_dir, data_dir, label_type, splits=("val", "test", "test_heldout")):
    """``evaluate.run_inference``'s twin for an arm checkpoint: the rows of ``manifest`` (``evaluate.load_split``)
    whose split is in ``splits``, scored with the eval transforms; returns ``evaluate.PREDICTION_COLUMNS``."""
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor

    import context_crops
    from evaluate import CLASSES, PREDICTION_COLUMNS

    _, model_cls = build_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir, use_fast=False)
    model = model_cls.from_pretrained(checkpoint_dir).to(device).eval()
    cfg = model.config
    label2id = {v: int(k) for k, v in cfg.id2label.items()}
    if set(label2id) != set(CLASSES):
        raise ValueError(f"unexpected id2label {cfg.id2label}")
    rows = manifest[manifest.split.isin(splits)]
    paths, side = None, None
    if cfg.context_arm in ("two_scale", "view_heatmap"):
        base_dir = os.path.join(data_dir, label_type)
        index = context_crops.load_index(base_dir, require_uids=rows.label_uid,
                                         crop_half_px=dict(zip(rows.label_uid.astype(str), rows.crop_half_px)))
        paths = dict(zip(index["label_uid"], zip(index["wide_path"], index["view_path"])))
        if cfg.index_sha256 and context_crops.index_sha256(base_dir) != cfg.index_sha256:
            print("NOTE: context/index.csv changed since training (rows added?); scoring with the current files",
                  file=sys.stderr)
    else:
        side = side_table(rows, cfg.age_reference_date, cfg.side_stats)
    inputs = ContextInputs(cfg.context_arm, processor, cfg.train_res, paths, side, cfg.heatmap_sigma_floor)
    probs = []
    for row in tqdm(rows.to_dict("records"), desc="inference"):
        uid = str(row["label_uid"])
        kw = {}
        if cfg.context_arm == "view_heatmap":
            kw["pixel_values"] = inputs.view(uid, row["crop_half_px"], train=False)[None]
        else:
            kw["pixel_values"] = processor(Image.open(row["crop_path"]).convert("RGB"),
                                           return_tensors="pt")["pixel_values"]
            if cfg.context_arm == "two_scale":
                kw["pixel_values_wide"] = inputs.wide(uid, train=False)[None]
            else:
                kw["side_features"] = inputs.side_vector(uid)[None]
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in kw.items()})
        probs.append(out.logits.softmax(-1).cpu().numpy()[0])
    probs = np.array(probs).reshape(-1, len(CLASSES))
    return pd.DataFrame({"label_uid": rows.label_uid.values, "split": rows.split.values,
                         "true": rows.label_class.values,
                         "p_correct": probs[:, label2id["correct"]], "p_incorrect": probs[:, label2id["incorrect"]]},
                        columns=PREDICTION_COLUMNS)


def build_parser():
    p = argparse.ArgumentParser(description="Score a context-arm checkpoint (issue #14).")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("predict", help="write <data-dir>/<label_type>/predictions_<tag>.csv for evaluate.py")
    pr.add_argument("--label_type", required=True)
    pr.add_argument("--data-dir", default=".", help="holds <label_type>/split.csv and <label_type>/context/")
    pr.add_argument("--checkpoint", required=True, help="a checkpoint-N directory of a --context-arm run")
    pr.add_argument("--tag", required=True, help="suffix of predictions_<tag>.csv (then evaluate.py --tag)")
    pr.add_argument("--splits", default="val,test,test_heldout", help="comma list of splits to score")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.tag.replace("-", "").replace("_", "").isalnum():
        build_parser().error("--tag must be letters, digits, '-' or '_' (it becomes part of file names)")
    import evaluate
    manifest = evaluate.load_split(args.label_type, args.data_dir)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    pred = run_inference(manifest, args.checkpoint, args.data_dir, args.label_type, splits)
    out = os.path.join(args.data_dir, args.label_type, f"predictions_{args.tag}.csv")
    pred.to_csv(out, index=False)
    print(f"wrote {out} ({len(pred)} rows); next: python evaluate.py --label_type {args.label_type} --data-dir "
          f"{args.data_dir} --skip-inference --tag {args.tag} --per-city")
    return 0


if __name__ == "__main__":
    sys.exit(main())
