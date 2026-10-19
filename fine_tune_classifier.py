"""Fine-tune a DINOv2-Large binary classifier ("correct" / "incorrect") for one Project Sidewalk label type.

Reads ``<data-dir>/<label_type>/split.csv`` (written by split_data.py; rows with ``split`` train and val), fine-tunes
``facebook/dinov2-large`` (Oquab et al., 2023, https://arxiv.org/abs/2304.07193) with the Hugging Face Trainer, and
keeps the checkpoint with the best validation macro-F1 (or loss, ``--best-metric loss``) under ``<label_type>/dinov2/`` (``--output-dir``). If
``split.csv`` is absent, the old ``<label_type>/ds/{train,val,test}/{correct,incorrect}`` image folders are
read instead, with a warning. The published models are at
https://huggingface.co/collections/projectsidewalk/project-sidewalk-validator-models-6797bd158d5e385e464dbe45
and are served by https://github.com/ProjectSidewalk/sidewalk-ai-api .

Preprocessing (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/5): train uses
``RandomResizedCrop(224, scale=(0.7, 1.0))``, ``RandomHorizontalFlip``, ``ToTensor`` and the checkpoint's ImageNet
``Normalize``; val uses the checkpoint's ``AutoImageProcessor`` itself (resize 256, centre-crop 224, normalise), which
is what evaluate.py and the API run, so checkpoint selection, evaluation and serving see identical tensors.

After training, ``split.csv``, ``deployment_priors.json`` (from ``<data-dir>``, if present) and ``training_run.json``
(seed, epochs, best checkpoint, package versions, and the tilt beta and depth encoder from
``generation_run.json``) are
written to the output dir, next to the checkpoints.

Resuming: a run that is interrupted (a klone ``ckpt`` preemption requeues the job) continues from its last saved
epoch. ``run_key.json`` in the output dir records the split's hash and the settings that shape training; checkpoints
written under a different key are never resumed, and the run stops instead of mixing them. The key covers all of
``split.csv``, so even a change to its test rows or ``hard`` column blocks a resume. The old imagefolder layout has no
``split.csv`` to hash, so resuming there is unguarded (a warning says so).

Usage::

    python fine_tune_classifier.py --label_type CurbRamp [--data-dir .] [--seed 42] [--epochs 25] \
        [--best-metric macro_f1|loss] [--num-workers 8]
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import warnings

import numpy as np
import pandas as pd

CHECKPOINT = "facebook/dinov2-large"
# Explicit and stable: the API reads index 0 as "correct". (imagefolder's alphabetical order gave the same.)
ID2LABEL = {0: "correct", 1: "incorrect"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def load_split_frames(label_type, data_dir="."):
    """``(train_df, val_df)`` with columns ``image`` (absolute crop path) and ``label`` (``LABEL2ID``), or None if
    ``<data_dir>/<label_type>/split.csv`` does not exist."""
    base = os.path.join(data_dir, label_type)
    path = os.path.join(base, "split.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not df["label_uid"].is_unique:
        raise ValueError(f"{path}: label_uid is not unique")
    df["image"] = [p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p)) for p in df["crop_path"]]
    df["label"] = df["label_class"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError(f"{path}: label_class must be one of {sorted(LABEL2ID)}")
    df["label"] = df["label"].astype(int)
    cols = ["image", "label"]
    return (df[df["split"] == "train"][cols].reset_index(drop=True),
            df[df["split"] == "val"][cols].reset_index(drop=True))


def generation_settings(label_type, data_dir="."):
    """The tilt beta and depth encoder the dataset was built with, from data_generation.py's
    ``<label_type>/generation_run.json`` (empty values if it is missing, e.g. an old imagefolder dataset)."""
    path = os.path.join(data_dir, label_type, "generation_run.json")
    run = {}
    if os.path.exists(path):
        with open(path) as f:
            run = json.load(f)
    return {"tilt_beta": run.get("tilt_beta"), "depth_encoder": run.get("depth_encoder")}


def default_num_workers():
    """Dataloader workers: the CPUs this process may use (the Slurm allocation, not the node), at most 8; 0 unless
    the start method is ``fork``, since spawn (macOS) cannot pickle the per-run transforms built in ``main``."""
    # The platform default is listed first; reading it this way does not fix the start method as a side effect.
    method = multiprocessing.get_start_method(allow_none=True) or multiprocessing.get_all_start_methods()[0]
    if method != "fork":
        return 0
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return min(8, max(cpus - 1, 0))  # leave one for the main process


def is_complete_checkpoint(path):
    """True if ``path`` is a ``checkpoint-N`` directory whose ``trainer_state.json`` parses: the Trainer writes that
    file after the weights, optimizer, scheduler and RNG state, so a checkpoint cut off mid-save fails this."""
    if not re.fullmatch(r"checkpoint-\d+", os.path.basename(os.path.normpath(path))):
        return False
    try:
        with open(os.path.join(path, "trainer_state.json")) as f:
            json.load(f)
        return True
    except (OSError, ValueError):
        return False


def run_key(label_type, data_dir, args):
    """What a resumed run must share with the run that wrote the checkpoints: the split file's hash and the
    settings that shape training. ``--num-workers`` and ``--allow-cpu`` may differ between attempts."""
    path = os.path.join(data_dir, label_type, "split.csv")
    split_sha256 = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            split_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {"label_type": label_type, "split_sha256": split_sha256, "seed": args.seed, "epochs": args.epochs,
            "best_metric": args.best_metric, "base_checkpoint": CHECKPOINT}


def resume_checkpoint(output_dir, key):
    """The checkpoint to resume from, or None for a fresh run (which writes ``run_key.json`` first).

    Only a complete checkpoint counts (``is_complete_checkpoint``). Raises SystemExit if checkpoints exist under a
    different (or no) key.

    Example::

        key = run_key("CurbRamp", "pilot", args)
        resume = resume_checkpoint("pilot/CurbRamp/dinov2", key)   # None on the first attempt
        trainer.train(resume_from_checkpoint=resume)
    """
    key_path = os.path.join(output_dir, "run_key.json")
    names = os.listdir(output_dir) if os.path.isdir(output_dir) else []
    ckpts = [d for d in names if is_complete_checkpoint(os.path.join(output_dir, d))]
    if not ckpts:
        os.makedirs(output_dir, exist_ok=True)
        with open(key_path, "w") as f:
            json.dump(key, f, indent=2)
            f.write("\n")
        return None
    old = None
    if os.path.exists(key_path):
        with open(key_path) as f:
            old = json.load(f)
    if old != key:
        if old is None:
            why = "it has no run_key.json"
        else:
            diff = {k: (old.get(k), key.get(k)) for k in sorted(set(old) | set(key)) if old.get(k) != key.get(k)}
            why = f"its run_key.json differs (saved, now): {diff}"
        raise SystemExit(f"{output_dir} holds checkpoints from another run: {why}. Move or delete it, or pass "
                         "--output-dir.")
    return os.path.join(output_dir, max(ckpts, key=lambda d: int(d.split("-")[1])))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_type', required=True, help='Specify the label type')
    parser.add_argument('--data-dir', default='.', help='root holding <label_type>/split.csv and deployment_priors.json')
    parser.add_argument('--output-dir', default=None, help='default <data-dir>/<label_type>/dinov2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--best-metric', choices=['macro_f1', 'loss'], default='macro_f1',
                        help='validation metric that picks the kept checkpoint (loss favours calibrated scores)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='dataloader worker processes (default: allocated CPUs - 1, at most 8; 0 off Linux); '
                             'with 0, decoding and augmentation starve the GPU')
    parser.add_argument('--allow-cpu', action='store_true',
                        help='run without CUDA (smoke tests only; a real run on the CPU would take weeks)')
    args = parser.parse_args(argv)

    import datasets
    import torch
    import transformers
    from datasets import Dataset, DatasetDict, load_dataset
    from sklearn.metrics import f1_score
    from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
    from transformers import AutoImageProcessor, Dinov2ForImageClassification, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            raise SystemExit("No CUDA device found. Fine-tuning requires a GPU; --allow-cpu is for smoke tests only.")
        warnings.warn("--allow-cpu: training on the CPU (smoke tests only).")
    transformers.set_seed(args.seed)
    label_type = args.label_type
    output_dir = args.output_dir or os.path.join(args.data_dir, label_type, "dinov2")
    if args.num_workers is None:
        args.num_workers = default_num_workers()
    key = run_key(label_type, args.data_dir, args)
    if key["split_sha256"] is None:
        warnings.warn("no split.csv: resuming from an existing checkpoint cannot check that the data is the same.")
    resume = resume_checkpoint(output_dir, key)  # before any slow setup

    frames = load_split_frames(label_type, args.data_dir)
    if frames is not None:
        train_df, val_df = frames
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df).cast_column("image", datasets.Image()),
            "validation": Dataset.from_pandas(val_df).cast_column("image", datasets.Image()),
        })
    else:
        warnings.warn(f"{os.path.join(args.data_dir, label_type, 'split.csv')} not found; falling back to the old "
                      "imagefolder layout <label_type>/ds/ (file-level split, no panorama grouping).")
        dataset = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir, label_type, "ds"))
        names = dataset["train"].features["label"].names
        if names != [ID2LABEL[0], ID2LABEL[1]]:
            raise ValueError(f"imagefolder classes {names} do not match {ID2LABEL}")

    # The slow (PIL) BitImageProcessor, as evaluate.py and sidewalk-ai-api (backend="pil") use; the fast torchvision
    # processor resizes slightly differently.
    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=False)
    crop = image_processor.crop_size
    size = (crop["height"], crop["width"]) if "height" in crop else crop["shortest_edge"]
    train_tf = Compose([
        RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

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
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    model = Dinov2ForImageClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # CPU only reachable with --allow-cpu

    def compute_metrics(pred):
        logits, labels = pred
        predictions = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, predictions, average="macro")}

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,

        remove_unused_columns=False,

        learning_rate=1e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        seed=args.seed,

        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,  # "loss" is the Trainer's eval_loss
        greater_is_better=args.best_metric != "loss",
        dataloader_num_workers=args.num_workers,
        push_to_hub=False,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=image_processor,
        compute_metrics=compute_metrics,
    )
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # Provenance next to the checkpoints, so a published model can be traced to its exact split and prior.
    for name, src in (("split.csv", os.path.join(args.data_dir, label_type, "split.csv")),
                      ("deployment_priors.json", os.path.join(args.data_dir, "deployment_priors.json"))):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, name))
    versions = {}
    for mod in ("torch", "torchvision", "transformers", "datasets", "numpy", "pandas", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - provenance is best-effort
            versions[mod] = None
    from data_generation import package_source
    versions["sidewalk-panorama-tools"] = package_source("sidewalk-panorama-tools")
    best = trainer.state.best_model_checkpoint
    run = {"label_type": label_type, "seed": args.seed, "epochs": args.epochs, "base_checkpoint": CHECKPOINT,
           "best_model_checkpoint": os.path.basename(best) if best else None,
           "best_metric": args.best_metric,
           "best_metric_value": trainer.state.best_metric, "id2label": ID2LABEL,
           **generation_settings(label_type, args.data_dir),
           "split_source": "split.csv" if frames is not None else "imagefolder", "versions": versions}
    with open(os.path.join(output_dir, "training_run.json"), "w") as f:
        json.dump(run, f, indent=2)
        f.write("\n")
    print(f"Best checkpoint: {best}; provenance in {output_dir}/training_run.json")


if __name__ == "__main__":
    main()
