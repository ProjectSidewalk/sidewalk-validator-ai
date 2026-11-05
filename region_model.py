"""A DINOv2 validator that is told where a label comes from: a learned country (or region) embedding concatenated to
the pooled image features before the classification head (issue #33).

The model is ``Dinov2ForImageClassification`` with one change. Its head reads ``[CLS token, mean of the patch tokens,
region embedding]`` instead of ``[CLS token, mean of the patch tokens]``. Row 0 of the embedding is the unknown
region (``<unk>``). During training each label's region id is replaced by ``<unk>`` with probability
``region_dropout`` (default 0.15), so the ``<unk>`` row learns to stand for "any country". A deployment in a country
the model never saw is scored as ``<unk>``, which is exactly the serving case for a new deployment.

The vocabulary is ``["<unk>"] + sorted(the train rows' countries or regions)`` and is saved in the checkpoint's
``config.json`` (``region_vocab``), so inference maps a city to the same id. The country of a city comes from
``regions.py`` (the ``/v3/api/cities`` listing, saved, because compute nodes have no network).

torch and transformers are imported inside functions, and the model classes are built by ``build_classes()``, so this
module imports on the CPU-only CI. The model is never registered with ``AutoModel``: ``evaluate.py``,
``publish_to_hf.py`` and sidewalk-ai-api cannot load it by accident. Load it with
``build_classes()[1].from_pretrained(path)``.

Usage (from ``fine_tune_classifier.py --region-conditioning country --cities-json cities.json``)::

    model, collate_fn = region_model.build_for_training(args, base, train_df, val_df, ID2LABEL, LABEL2ID, collate_fn)
"""
import functools
import hashlib
import json
import warnings

import regions

UNK = "<unk>"
DEFAULT_DIM = 16
DEFAULT_DROPOUT = 0.15


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers (no torch)
# ---------------------------------------------------------------------------------------------------------------------

def region_value(city, level, table):
    """The country or region (``level``) of one ``city_id``; ``regions.UNKNOWN`` when it has none.

    >>> t = regions.country_table([], {})
    >>> region_value("cdmx", "country", t), region_value("cdmx", "region", t)
    ('mexico', 'latin-america')
    """
    if level not in regions.LEVELS:
        raise ValueError(f"level must be one of {regions.LEVELS}")
    country = regions.country_of(city, table)
    return country if level == "country" else regions.region_of(country)


def region_vocab_from_split(train_df, level, table):
    """``["<unk>"] + sorted(set of the train rows' countries or regions)``, from the ``city`` column. Raises
    ``ValueError`` when a train row's deployment has no country (``regions.assign``). ``regions.UNKNOWN`` (a
    country without a region, at the region level) is never a vocabulary entry: those rows are ``<unk>`` (row 0),
    the row every unseen country is scored with, not a learned region of their own.

    >>> import pandas as pd
    >>> t = regions.country_table([], {})
    >>> region_vocab_from_split(pd.DataFrame({"city": ["spgg", "amsterdam", "cdmx"]}), "country", t)
    ['<unk>', 'mexico', 'netherlands']
    >>> t2 = regions.country_table([{"city_id": "x", "country_id": "atlantis"}], {})
    >>> region_vocab_from_split(pd.DataFrame({"city": ["x", "cdmx"]}), "region", t2)
    ['<unk>', 'latin-america']
    """
    values = regions.assign(train_df[["city"]], table)[level]
    return [UNK] + sorted(set(values) - {regions.UNKNOWN, UNK})


def region_ids(values, vocab):
    """Index of each value in ``vocab``; a value outside it, or ``regions.UNKNOWN``, is 0 (``<unk>``).

    >>> region_ids(["b", "zz", "a", "unknown"], ["<unk>", "a", "b", "unknown"])
    [2, 0, 1, 0]
    """
    index = {v: i for i, v in enumerate(vocab)}
    return [0 if v == regions.UNKNOWN else index.get(v, 0) for v in values]


def city_region_ids(cities, vocab, level, table):
    """``{city_id: region id}`` for every distinct city: its country or region's index in ``vocab``, 0 when unknown
    or outside the vocabulary.

    >>> t = regions.country_table([], {})
    >>> city_region_ids(["cdmx", "amsterdam", "nowhere"], ["<unk>", "mexico"], "country", t)
    {'amsterdam': 0, 'cdmx': 1, 'nowhere': 0}
    """
    uniq = sorted(set(str(c) for c in cities))
    return dict(zip(uniq, region_ids([region_value(c, level, table) for c in uniq], vocab)))


def vocab_sha256(vocab):
    """sha256 of the vocabulary as JSON (for provenance)."""
    return hashlib.sha256(json.dumps(list(vocab)).encode("utf-8")).hexdigest()


def wrap_collate(collate_fn, city_to_id):
    """A collate function that adds ``region_ids`` (one id per example, from its ``city``) to ``collate_fn``'s
    batch; works on the hard and the soft-target collates alike. An example's city missing from ``city_to_id`` is
    ``<unk>``."""
    def collate(examples):
        import torch
        batch = collate_fn(examples)
        batch["region_ids"] = torch.tensor([city_to_id.get(str(e["city"]), 0) for e in examples], dtype=torch.long)
        return batch
    return collate


# ---------------------------------------------------------------------------------------------------------------------
# The model (torch + transformers)
# ---------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def build_classes():
    """``(RegionConditionedDinov2Config, RegionConditionedDinov2)``, built once (transformers imported here)."""
    import torch
    from torch import nn
    from transformers import Dinov2Config, Dinov2Model, Dinov2PreTrainedModel
    from transformers.modeling_outputs import ImageClassifierOutput

    class RegionConditionedDinov2Config(Dinov2Config):
        model_type = "region_dinov2"

        def __init__(self, region_vocab=None, region_dim=DEFAULT_DIM, region_dropout=DEFAULT_DROPOUT,
                     region_level="country", **kwargs):
            super().__init__(**kwargs)
            self.region_vocab = list(region_vocab) if region_vocab else [UNK]
            self.region_dim = int(region_dim)
            self.region_dropout = float(region_dropout)
            self.region_level = region_level

    class RegionConditionedDinov2(Dinov2PreTrainedModel):
        config_class = RegionConditionedDinov2Config
        # The Trainer then divides the loss by the gradient-accumulation steps, as for Dinov2ForImageClassification
        # with the hard target; the forward below takes no **kwargs.
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.dinov2 = Dinov2Model(config)
            self.region_embedding = nn.Embedding(len(config.region_vocab), config.region_dim)
            self.classifier = nn.Linear(2 * config.hidden_size + config.region_dim, config.num_labels)
            self.post_init()
            nn.init.normal_(self.region_embedding.weight, std=config.initializer_range)

        def region_input(self, region_ids, batch_size, device):
            """The ids the head sees: zeros (``<unk>``) when none are given; in training mode, each id replaced by
            0 with probability ``region_dropout``."""
            if region_ids is None:
                return torch.zeros(batch_size, dtype=torch.long, device=device)
            ids = region_ids.to(device=device, dtype=torch.long)
            p = self.config.region_dropout
            if self.training and p > 0:
                ids = torch.where(torch.rand(ids.shape, device=device) < p, torch.zeros_like(ids), ids)
            return ids

        def forward(self, pixel_values=None, region_ids=None, labels=None):
            seq = self.dinov2(pixel_values=pixel_values).last_hidden_state
            pooled = torch.cat([seq[:, 0], seq[:, 1:].mean(dim=1)], dim=1)  # Dinov2ForImageClassification's pooling
            ids = self.region_input(region_ids, seq.shape[0], seq.device)
            logits = self.classifier(torch.cat([pooled, self.region_embedding(ids)], dim=1))
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))
            return ImageClassifierOutput(loss=loss, logits=logits)

        @classmethod
        def from_base(cls, base, vocab, region_dim=DEFAULT_DIM, region_dropout=DEFAULT_DROPOUT, level="country",
                      id2label=None, label2id=None):
            """A new model whose backbone is ``base``'s (``facebook/dinov2-large``, or a ``checkpoint-N`` directory of
            either class) and whose head and region embedding are freshly initialised."""
            base_cfg = Dinov2Config.from_pretrained(base).to_dict()
            for k in ("model_type", "architectures", "transformers_version", "id2label", "label2id", "num_labels",
                      "region_vocab", "region_dim", "region_dropout", "region_level"):
                base_cfg.pop(k, None)
            cfg = RegionConditionedDinov2Config(region_vocab=vocab, region_dim=region_dim,
                                                region_dropout=region_dropout, region_level=level,
                                                id2label=id2label, label2id=label2id, **base_cfg)
            model = cls(cfg)
            backbone = Dinov2Model.from_pretrained(base)
            model.dinov2.load_state_dict(backbone.state_dict())
            return model

    return RegionConditionedDinov2Config, RegionConditionedDinov2


def build_for_training(args, base, train_df, val_df, id2label, label2id, collate_fn):
    """The region-conditioned model and its collate for ``fine_tune_classifier.py``: ``(model, collate_fn)``.

    ``train_df``/``val_df`` need a ``city`` column (``load_split_frames(..., extra_columns=("city",))``). The
    vocabulary comes from the train rows at ``args.region_conditioning`` (``country`` or ``region``); val cities
    outside it are ``<unk>`` (counted in a warning). ``args.cities_json`` is the saved listing.
    """
    if train_df is None or "city" not in train_df.columns:
        raise SystemExit("--region-conditioning needs split.csv with a city column (not the imagefolder layout)")
    level = args.region_conditioning
    table = regions.country_table(regions.load_listing(args.cities_json))
    vocab = region_vocab_from_split(train_df, level, table)
    city_to_id = city_region_ids(list(train_df["city"]) + list(val_df["city"]), vocab, level, table)
    n_unk = int(sum(city_to_id[str(c)] == 0 for c in val_df["city"]))
    if n_unk:
        warnings.warn(f"{n_unk} of {len(val_df)} val rows are in a {level} outside the training vocabulary; they "
                      "are scored as <unk>")
    n_unk_train = int(sum(city_to_id[str(c)] == 0 for c in train_df["city"]))
    if n_unk_train:
        warnings.warn(f"{n_unk_train} of {len(train_df)} train rows have no {level} (a country without a region; "
                      f"set {regions.OVERRIDES_ENV}); they train the <unk> row")
    # Counts only: the vocabulary itself holds country ids whose deployments may all be private (it is in the
    # checkpoint's config.json, which stays on klone scratch and is never published as is).
    print(f"region conditioning at the {level} level: {len(vocab) - 1} {level} ids plus <unk> "
          f"(vocabulary sha256 {vocab_sha256(vocab)[:12]})")
    _, model_cls = build_classes()
    model = model_cls.from_base(base, vocab, args.region_dim, args.region_dropout, level, id2label, label2id)
    return model, wrap_collate(collate_fn, city_to_id)


def run_inference(manifest, checkpoint_dir, splits, table, force_unk=False):
    """Score the rows of ``manifest`` whose split is in ``splits`` with a region-conditioned checkpoint; returns the
    columns of ``evaluate.PREDICTION_COLUMNS``. Each row's region id comes from its ``city`` through the
    checkpoint's own ``region_vocab`` and ``region_level``; ``force_unk`` scores every row as ``<unk>`` (the
    embedding ablation). Same processor and checks as ``evaluate.run_inference``."""
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoImageProcessor

    from evaluate import CLASSES, PREDICTION_COLUMNS

    _, model_cls = build_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir, use_fast=False)
    model = model_cls.from_pretrained(checkpoint_dir).to(device).eval()
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    if set(label2id) != set(CLASSES):
        raise ValueError(f"unexpected id2label {model.config.id2label}")
    rows = manifest[manifest.split.isin(splits)]
    ids = city_region_ids(rows.city, model.config.region_vocab, model.config.region_level, table)
    probs = []
    for path, city in tqdm(zip(rows.crop_path, rows.city), total=len(rows), desc="inference"):
        image = Image.open(path).convert("RGB")
        inputs = image_processor(image, return_tensors="pt").to(device)
        rid = torch.tensor([0 if force_unk else ids[str(city)]], device=device)
        with torch.no_grad():
            probs.append(model(pixel_values=inputs["pixel_values"], region_ids=rid).logits.softmax(-1).cpu().numpy()[0])
    probs = np.array(probs).reshape(-1, len(CLASSES))
    return pd.DataFrame({"label_uid": rows.label_uid.values, "split": rows.split.values,
                         "true": rows.label_class.values,
                         "p_correct": probs[:, label2id["correct"]], "p_incorrect": probs[:, label2id["incorrect"]]},
                        columns=PREDICTION_COLUMNS)
