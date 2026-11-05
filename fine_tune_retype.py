"""Fine-tune one DINOv2-Large over the eight label types (``retype.CLASSES``): validation as re-typing (issue #13).

Reads ``retype_split.csv`` (``python retype.py assemble``), trains on its ``train`` rows and keeps the checkpoint with
the best validation ``--best-metric`` (default ``auc_claimed``: the mean, over the five validator types weighted by
their val rows, of the AUC of ``p(claimed)`` for correct vs incorrect, which is the verdict the API would serve).
The recipe is round 3's (``fine_tune_classifier.py``): ``facebook/dinov2-large``, batch 8 x accumulation 4, lr 1e-5,
warm-up 0.1, train augmentation ``RandomResizedCrop(224, scale=(0.7, 1.0))`` + flip, val preprocessing = the
checkpoint's slow ``AutoImageProcessor`` (what evaluate.py and sidewalk-ai-api run). Only the head and the loss
differ.

Loss (``retype_loss``):

- a ``correct`` row is a positive of its type: cross-entropy on ``target_type``, weighted per class
  (``retype.class_weights``, ``--class-weight-power``; 0 = off);
- an ``incorrect`` row only says "not the claimed type". ``--negatives complement`` (default) adds
  ``-log(1 - p(claimed))`` (complementary-label learning: the mass may go to any other class, ``Other`` included);
  ``--negatives other`` is the ablation that treats it as an ``Other`` positive.

Resuming works as in fine_tune_classifier.py (``run_key.json`` in the output dir, ``resume_checkpoint``): the key holds
the split file's hash and every setting that shapes training, including ``--negatives``, so a requeued job continues
from its last complete checkpoint and a run under other settings is refused.

Usage::

    python fine_tune_retype.py --split OUT/retype_split.csv --crop-root DATA --output-dir RUNS/retype/complement \\
        [--seed 42] [--epochs 12] [--best-metric auc_claimed|macro_f1|loss] [--negatives complement|other] \\
        [--class-weight-power 0.5] [--cap-per-class N] [--num-workers 7] [--allow-cpu]
"""
import argparse
import json
import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

import fine_tune_classifier as ftc
import retype as rt

TRAIN_BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
CROP_SCALE = (0.7, 1.0)
IGNORE = -100  # the target of an incorrect row: no positive class


def retype_run_key(args):
    """``fine_tune_classifier.run_key`` for the split file (its sha256, seed, epochs, best metric, base checkpoint)
    plus what this script adds: ``negatives``, ``class_weight_power``, ``cap_per_class`` and the class list.

    >>> import tempfile; d = tempfile.mkdtemp(); p = os.path.join(d, "s.csv"); _ = open(p, "w").write("x\\n")
    >>> a = SimpleNamespace(split=p, seed=1, epochs=2, best_metric="loss", negatives="complement",
    ...                     class_weight_power=0.5, cap_per_class=None)
    >>> k = retype_run_key(a); k["label_type"], k["negatives"], len(k["classes"])
    ('retype', 'complement', 8)
    """
    base = ftc.run_key("retype", os.path.dirname(os.path.abspath(args.split)),
                       SimpleNamespace(seed=args.seed, epochs=args.epochs, best_metric=args.best_metric,
                                       split=args.split))
    return {**base, "negatives": args.negatives, "class_weight_power": args.class_weight_power,
            "cap_per_class": args.cap_per_class, "classes": list(rt.CLASSES)}


def load_frames(split_path, crop_root, cap_per_class=None, seed=42):
    """``(train_df, val_df, dropped_by_cap)``: columns ``image`` (absolute crop path), ``label`` (target index, or
    ``IGNORE`` for an incorrect row), ``claimed`` (claimed index) and ``is_correct``.

    >>> import tempfile; d = tempfile.mkdtemp(); p = os.path.join(d, "s.csv")
    >>> pd.DataFrame({"row_uid": ["c:1@CurbRamp", "c:2@Obstacle"], "split": ["train", "val"],
    ...     "claimed_type": ["CurbRamp", "Obstacle"], "target_type": ["CurbRamp", ""], "is_correct": [1, 0],
    ...     "crop_path": ["CurbRamp/a.webp", "/abs/b.webp"]}).to_csv(p, index=False)
    >>> tr, va, _ = load_frames(p, "/root")
    >>> tr.values.tolist(), va.values.tolist()
    ([['/root/CurbRamp/a.webp', 0, 0, 1]], [['/abs/b.webp', -100, 2, 0]])
    """
    df = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    if not df["row_uid"].is_unique:
        raise ValueError(f"{split_path}: row_uid is not unique")
    df["is_correct"] = df["is_correct"].astype(int)
    df, dropped = rt.cap_per_class(df, cap_per_class, seed)
    df = df[df["split"].isin(["train", "val"])].copy()
    df["image"] = [p if os.path.isabs(p) else os.path.abspath(os.path.join(crop_root, p)) for p in df["crop_path"]]
    df["claimed"] = rt.claimed_index(df["claimed_type"])
    df["label"] = [rt.LABEL2ID[t] if c == 1 else IGNORE for t, c in zip(df["target_type"], df["is_correct"])]
    cols = ["image", "label", "claimed", "is_correct"]
    return (df[df["split"] == "train"][cols].reset_index(drop=True),
            df[df["split"] == "val"][cols].reset_index(drop=True), dropped)


def train_counts(train_df):
    """Positive train rows per class (all eight, 0 included)."""
    pos = train_df[train_df["label"] != IGNORE]["label"].value_counts()
    return {c: int(pos.get(i, 0)) for i, c in rt.ID2LABEL.items()}


def retype_loss(logits, labels, claimed, is_correct, class_weights=None, negatives="complement",
                num_items_in_batch=None):
    """Per-row loss summed and divided by the batch size (or ``num_items_in_batch`` when given).

    Positives (``is_correct == 1``): ``w[label] * CE(logits, label)``. Negatives: ``complement`` =
    ``-log(1 - softmax(logits)[claimed])`` = ``-log sum_{j != claimed} p_j``, computed exactly as
    ``logsumexp(logits) - logsumexp(logits with the claimed column masked to -inf)`` (no clamp, so a confidently
    wrong row keeps its gradient at any logit gap), or ``other`` = ``CE(logits, Other)``. Negatives carry weight 1.0
    in both legs: ``class_weights`` come from positive counts only, so weighting the ``other`` leg's negatives by
    ``w[Other]`` would let a handful of Other positives rescale every negative row and break the comparison with the
    ``complement`` leg. ``class_weights``: a length-8 tensor or None (all 1).

    Example::

        loss = retype_loss(model(pixel_values=x).logits, labels, claimed, is_correct, w, "complement")
    """
    import torch
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    n_cls = logits.shape[-1]
    w = class_weights.to(logp) if class_weights is not None else torch.ones(n_cls, dtype=logp.dtype,
                                                                             device=logp.device)
    pos = is_correct.bool()
    per_row = torch.zeros(logits.shape[0], dtype=logp.dtype, device=logp.device)
    if pos.any():
        t = labels[pos].long()
        per_row[pos] = -logp[pos].gather(1, t[:, None]).squeeze(1) * w[t]
    neg = ~pos
    if neg.any():
        if negatives == "complement":
            lg = logits[neg]
            claimed_col = torch.nn.functional.one_hot(claimed[neg].long(), n_cls).bool()
            masked = lg.masked_fill(claimed_col, float("-inf"))
            per_row[neg] = torch.logsumexp(lg, dim=-1) - torch.logsumexp(masked, dim=-1)
        elif negatives == "other":
            other = rt.LABEL2ID[rt.OTHER]
            per_row[neg] = -logp[neg][:, other]  # weight 1.0, not w[Other] (see the docstring)
        else:
            raise ValueError(f"negatives must be complement or other, got {negatives!r}")
    denom = num_items_in_batch if num_items_in_batch is not None else logits.shape[0]
    return per_row.sum() / denom


def _softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_retype_metrics(logits, labels, claimed, is_correct):
    """Validation metrics: ``top1_acc`` and ``macro_f1`` of the type head on positive rows (``labels != -100``;
    F1 over the classes present), ``auc_claimed_<type>`` of ``p(claimed)`` (correct vs incorrect) per claimed type
    with both classes, and ``auc_claimed`` = the mean of those AUCs over the five ``retype.VALIDATOR_TYPES`` only,
    weighted by each type's val row count (0.0 when none has both classes, so checkpoint selection still has a
    number). The weighting keeps a small, noisy val type (NoSidewalk has a few dozen rows) from moving checkpoint
    selection as much as CurbRamp; the extras (NoSidewalk, Signal, Other) are reported but do not select, since no
    deployed validator serves them.

    >>> lg = np.log(np.array([[.8, .1, .1] + [1e-9] * 5, [.2, .7, .1] + [1e-9] * 5, [.6, .3, .1] + [1e-9] * 5]))
    >>> m = compute_retype_metrics(lg, [0, -100, 1], [0, 0, 1], [1, 0, 1])
    >>> m["top1_acc"], m["auc_claimed_CurbRamp"], m["auc_claimed"]
    (0.5, 1.0, 1.0)
    """
    from sklearn.metrics import f1_score, roc_auc_score
    p = _softmax(logits)
    labels, claimed, is_correct = (np.asarray(v).astype(int).ravel() for v in (labels, claimed, is_correct))
    pos = labels != IGNORE
    out = {}
    if pos.any():
        pred = p[pos].argmax(axis=1)
        out["top1_acc"] = float((pred == labels[pos]).mean())
        present = sorted(set(labels[pos]))
        out["macro_f1"] = float(f1_score(labels[pos], pred, labels=present, average="macro", zero_division=0))
    else:
        out["top1_acc"] = out["macro_f1"] = 0.0
    pc = p[np.arange(len(p)), claimed]
    aucs, ns = [], []
    for i, c in rt.ID2LABEL.items():
        m = claimed == i
        if m.any() and 0 < is_correct[m].sum() < m.sum():
            out[f"auc_claimed_{c}"] = float(roc_auc_score(is_correct[m], pc[m]))
            if c in rt.VALIDATOR_TYPES:
                aucs.append(out[f"auc_claimed_{c}"])
                ns.append(int(m.sum()))
    out["auc_claimed"] = float(np.average(aucs, weights=ns)) if aucs else 0.0
    return out


def trainer_metrics(pred):
    """``compute_retype_metrics`` from a Trainer ``EvalPrediction`` whose ``label_ids`` are (labels, claimed,
    is_correct), the order of ``LABEL_NAMES``."""
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels, claimed, is_correct = pred.label_ids
    return compute_retype_metrics(logits, labels, claimed, is_correct)


LABEL_NAMES = ["labels", "claimed", "is_correct"]


def retype_trainer(trainer_cls, class_weights, negatives):
    """A subclass of ``trainer_cls`` whose ``compute_loss`` is ``retype_loss`` on every batch (train and val).

    The Trainer's own ``num_items_in_batch`` counts ``labels != -100``, i.e. positives only, which would mis-scale
    the negatives, so it is not used: the loss is the batch mean, and ``model_accepts_loss_kwargs = False`` makes the
    Trainer divide by the accumulation steps itself."""

    class RetypeTrainer(trainer_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            labels, claimed, is_correct = (inputs.pop(k) for k in LABEL_NAMES)
            outputs = model(pixel_values=inputs["pixel_values"])
            loss = retype_loss(outputs.logits, labels, claimed, is_correct, class_weights, negatives)
            return (loss, outputs) if return_outputs else loss

    return RetypeTrainer


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", required=True, help="retype_split.csv (retype.py assemble)")
    p.add_argument("--crop-root", required=True, help="root the relative crop_paths resolve against")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--best-metric", choices=["auc_claimed", "macro_f1", "loss"], default="auc_claimed")
    p.add_argument("--negatives", choices=["complement", "other"], default="complement",
                   help="loss for incorrect rows: complement = -log(1 - p(claimed)); other = an Other positive")
    p.add_argument("--class-weight-power", type=float, default=0.5, help="0 = unweighted positives")
    p.add_argument("--cap-per-class", type=int, default=None, help="cap each class's train positives (seeded)")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true", help="smoke tests only")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.split):
        raise SystemExit(f"--split {args.split} not found")

    import datasets
    import torch
    import transformers
    from datasets import Dataset, DatasetDict
    from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
    from transformers import AutoImageProcessor, Dinov2ForImageClassification, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("No CUDA device found. Fine-tuning requires a GPU; --allow-cpu is for smoke tests only.")
        warnings.warn("--allow-cpu: training on the CPU (smoke tests only).")
    transformers.set_seed(args.seed)
    if args.num_workers is None:
        args.num_workers = ftc.default_num_workers()
    key = retype_run_key(args)
    resume = ftc.resume_checkpoint(args.output_dir, key)  # before any slow setup

    train_df, val_df, dropped = load_frames(args.split, args.crop_root, args.cap_per_class, args.seed)
    counts = train_counts(train_df)
    weights = rt.class_weights(counts, args.class_weight_power)
    print(f"train {len(train_df)} rows ({sum(counts.values())} positives: {counts}); val {len(val_df)}; "
          f"class weights {weights}" + (f"; capped {dropped}" if dropped else ""))
    w_tensor = torch.tensor([weights[c] for c in rt.CLASSES], dtype=torch.float32)

    # From plain lists, so the column types do not depend on pandas' string dtype (pandas 3 makes it Arrow-backed).
    def to_dataset(df):
        return Dataset.from_dict({c: df[c].tolist() for c in df.columns}).cast_column("image", datasets.Image())
    dataset = DatasetDict({"train": to_dataset(train_df), "validation": to_dataset(val_df)})
    # As fine_tune_classifier.py: the slow (PIL) processor, which evaluate.py and sidewalk-ai-api use.
    image_processor = AutoImageProcessor.from_pretrained(ftc.CHECKPOINT, use_fast=False)
    crop = image_processor.crop_size
    size = (crop["height"], crop["width"]) if "height" in crop else crop["shortest_edge"]
    train_tf = Compose([RandomResizedCrop(size, scale=CROP_SCALE, antialias=True), RandomHorizontalFlip(), ToTensor(),
                        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])

    def train_transforms(examples):
        examples["pixel_values"] = [train_tf(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    def val_transforms(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = list(image_processor(images, return_tensors="pt")["pixel_values"])
        del examples["image"]
        return examples

    dataset["train"] = dataset["train"].with_transform(train_transforms)
    dataset["validation"] = dataset["validation"].with_transform(val_transforms)

    def collate_fn(examples):
        return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]),
                "labels": torch.tensor([e["label"] for e in examples]),
                "claimed": torch.tensor([e["claimed"] for e in examples]),
                "is_correct": torch.tensor([e["is_correct"] for e in examples])}

    model = Dinov2ForImageClassification.from_pretrained(ftc.CHECKPOINT, num_labels=len(rt.CLASSES),
                                                         id2label=rt.ID2LABEL, label2id=rt.LABEL2ID,
                                                         ignore_mismatched_sizes=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        output_dir=args.output_dir, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        logging_steps=10, remove_unused_columns=False, label_names=LABEL_NAMES,
        learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        num_train_epochs=args.epochs, warmup_ratio=WARMUP_RATIO, seed=args.seed,
        load_best_model_at_end=True, metric_for_best_model=args.best_metric,
        greater_is_better=args.best_metric != "loss", dataloader_num_workers=args.num_workers,
        push_to_hub=False, report_to="none", use_cpu=not torch.cuda.is_available())
    trainer = retype_trainer(Trainer, w_tensor, args.negatives)(
        model=model, args=training_args, data_collator=collate_fn, train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], processing_class=image_processor, compute_metrics=trainer_metrics)
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    shutil.copy(args.split, os.path.join(args.output_dir, "retype_split.csv"))
    versions = {}
    for mod in ("torch", "torchvision", "transformers", "datasets", "numpy", "pandas", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - provenance is best-effort
            versions[mod] = None
    from data_generation import package_source
    versions["sidewalk-panorama-tools"] = package_source("sidewalk-panorama-tools")
    assemble_json = os.path.join(os.path.dirname(os.path.abspath(args.split)), "assemble.json")
    source_runs = {}
    if os.path.exists(assemble_json):
        with open(assemble_json) as f:
            source_runs = {t: v["sha256"] for t, v in json.load(f).get("types", {}).items()}
    best = trainer.state.best_model_checkpoint
    run = {"label_type": "retype", "seed": args.seed, "epochs": args.epochs, "base_checkpoint": ftc.CHECKPOINT,
           "best_model_checkpoint": os.path.basename(best) if best else None, "best_metric": args.best_metric,
           "best_metric_value": trainer.state.best_metric, "id2label": rt.ID2LABEL,
           "split_path": os.path.abspath(args.split), "split_sha256": key["split_sha256"],
           "classes": list(rt.CLASSES), "negatives": args.negatives, "class_weight_power": args.class_weight_power,
           "class_weights": weights, "cap_per_class": args.cap_per_class, "capped": dropped,
           "n_train_by_class": counts, "n_train_negatives": int((train_df["label"] == IGNORE).sum()),
           "crop_root": os.path.abspath(args.crop_root), "source_runs": source_runs, "versions": versions}
    with open(os.path.join(args.output_dir, "training_run.json"), "w") as f:
        json.dump(run, f, indent=2)
        f.write("\n")
    print(f"Best checkpoint: {best}; provenance in {args.output_dir}/training_run.json")


if __name__ == "__main__":
    main()
