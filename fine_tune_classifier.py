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

Training target (``--target``, https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/12):

- ``hard`` (default): the 0/1 ``label_class``, with the model's own cross-entropy. Without ``--target`` the run, its
  ``run_key.json`` and its ``training_run.json`` are exactly what they were before the option existed.
- ``soft``: each train row's target probability of "correct" is the smoothed human agreement rate
  ``(human_agree + 1) / (human_agree + human_disagree + 2)``, from split.csv's ``human_agree``/``human_disagree``
  columns (data_generation.py's manifest, carried through by split_data.py; missing columns are an error).
- ``weighted``: the same formula on the reliability-weighted counts ``w_agree``/``w_disagree`` from ``--vote-weights``
  (validator_reliability.py's ``vote_weights.csv``), joined on ``label_uid``; a train row without one is an error, and
  so is a row whose ``n_votes`` differs from split.csv's ``human_agree + human_disagree`` (the weights were built from
  a different rawLabels snapshot than the manifest).

A non-hard target writes to ``<label_type>/dinov2-<target>`` by default, never to ``<label_type>/dinov2`` where
evaluate.py and publish_to_hf.py look for the production model; score it with
``evaluate.py --checkpoint <dir>/checkpoint-N --tag <target>``.

Soft targets train with soft-label cross-entropy against the two logits (``soft_target_loss``). Validation stays hard:
val rows carry only ``label_class``, so ``eval_loss`` is the plain cross-entropy and ``macro_f1`` the hard-label
F1, and checkpoint selection means the same thing for every target. A non-default target, and the vote-weights
file's hash, are recorded in ``run_key.json`` (so a resume never mixes targets) and in ``training_run.json``.

Usage::

    python fine_tune_classifier.py --label_type CurbRamp [--data-dir .] [--seed 42] [--epochs 25] \
        [--best-metric macro_f1|loss] [--num-workers 8] \
        [--target hard|soft|weighted] [--vote-weights reliability/vote_weights.csv]
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


def smoothed_rate(agree, disagree):
    """``(agree + 1) / (agree + disagree + 2)``: the smoothed share of Agree votes (the soft target for "correct").

    >>> smoothed_rate(3, 0), smoothed_rate(0, 2), smoothed_rate(0.0, 0.0)
    (0.8, 0.25, 0.5)
    """
    return (agree + 1) / (agree + disagree + 2)


def soft_targets(df, target, vote_weights=None, path="split.csv"):
    """Per-row target probability of "correct" for ``target`` ``soft`` or ``weighted``, aligned with ``df``.

    ``soft`` reads ``df``'s ``human_agree``/``human_disagree``; ``weighted`` joins ``vote_weights`` (a DataFrame with
    ``label_uid, w_agree, w_disagree`` and, checked against ``df``'s vote columns when both have them, ``n_votes``)
    on ``label_uid``. Missing columns or rows, or a vote-count mismatch, raise ValueError.

    >>> d = pd.DataFrame({"label_uid": ["a:1", "a:2"], "human_agree": [3, 0], "human_disagree": [0, 2]})
    >>> soft_targets(d, "soft").tolist()
    [0.8, 0.25]
    >>> w = pd.DataFrame({"label_uid": ["a:2", "a:1", "b:9"], "w_agree": [0.0, 2.0, 1.0],
    ...                   "w_disagree": [1.5, 0.0, 0.0], "n_votes": [2, 3, 1]})
    >>> soft_targets(d, "weighted", w).tolist()
    [0.75, 0.2857142857142857]
    """
    if target == "soft":
        missing = {"human_agree", "human_disagree"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} has no {sorted(missing)} column(s), which --target soft needs; rebuild it with "
                             "split_data.py from a data_generation.py manifest")
        agree, disagree = df["human_agree"], df["human_disagree"]
    elif target == "weighted":
        if vote_weights is None:
            raise ValueError("--target weighted needs --vote-weights")
        missing = {"label_uid", "w_agree", "w_disagree"} - set(vote_weights.columns)
        if missing:
            raise ValueError(f"vote weights have no {sorted(missing)} column(s)")
        cols = [c for c in ("label_uid", "w_agree", "w_disagree", "n_votes") if c in vote_weights.columns]
        w = df[["label_uid"]].merge(vote_weights[cols], on="label_uid", how="left", validate="one_to_one")
        absent = w["w_agree"].isna() | w["w_disagree"].isna()
        if absent.any():
            raise ValueError(f"{int(absent.sum())} of {len(w)} rows of {path} have no vote weights (e.g. "
                             f"{w.loc[absent, 'label_uid'].head(3).tolist()}); rebuild vote_weights.csv with "
                             "validator_reliability.py from the rawLabels the manifest was built from")
        if "n_votes" in w.columns and {"human_agree", "human_disagree"} <= set(df.columns):
            human = (df["human_agree"] + df["human_disagree"]).to_numpy()
            bad = w["n_votes"].to_numpy() != human
            if bad.any():
                raise ValueError(f"{int(bad.sum())} of {len(w)} rows of {path} have n_votes in the vote weights != "
                                 f"human_agree + human_disagree (e.g. {w.loc[bad, 'label_uid'].head(3).tolist()}): "
                                 "the weights come from a different rawLabels snapshot than the manifest, or a "
                                 "validator voted twice on a label (validator_reliability.py keeps their last vote; "
                                 "data_generation.human_vote_counts counts both)")
        else:
            warnings.warn(f"vote-count snapshot check skipped: {path} lacks human_agree/human_disagree or the vote "
                          "weights lack n_votes, so weights from a different rawLabels snapshot cannot be detected")
        agree, disagree = w["w_agree"], w["w_disagree"]
    else:
        raise ValueError(f"no soft target for target {target!r}")
    out = smoothed_rate(agree.astype(float), disagree.astype(float))
    if out.isna().any():
        raise ValueError(f"{path}: vote counts must not be empty for --target {target}")
    return pd.Series(out.to_numpy(), index=df.index)


def soft_target_loss(logits, p_correct, num_items_in_batch=None):
    """Soft-label cross-entropy: ``-(p * log softmax(logits)[correct] + (1 - p) * log softmax(logits)[incorrect])``,
    with the class order of ``ID2LABEL``. Mean over the batch, or the sum divided by ``num_items_in_batch`` when the
    Trainer passes one (gradient accumulation across batches). With 0/1 targets it equals the hard cross-entropy.

    Example::

        loss = soft_target_loss(model(pixel_values=x).logits, torch.tensor([0.8, 0.25]))
    """
    import torch
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = p_correct.to(logp.dtype)
    target = torch.zeros_like(logp)
    target[:, LABEL2ID["correct"]] = p
    target[:, LABEL2ID["incorrect"]] = 1 - p
    per_row = -(target * logp).sum(dim=-1)
    if num_items_in_batch is not None:
        return per_row.sum() / num_items_in_batch
    return per_row.mean()


def soft_target_trainer(trainer_cls):
    """A subclass of ``trainer_cls`` (the transformers Trainer) whose ``compute_loss`` uses ``soft_target_loss`` on
    batches that carry ``soft_targets``; other batches (validation) get the parent's hard loss."""

    class SoftTargetTrainer(trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "soft_targets" not in inputs:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            inputs = dict(inputs)
            p = inputs.pop("soft_targets")
            inputs.pop("labels", None)  # the model would add its own hard loss
            outputs = model(**inputs)
            # As the Trainer does for a model's own loss: divide by the items in the accumulated batch only when it
            # will not itself divide by the accumulation steps.
            n = num_items_in_batch if getattr(self, "model_accepts_loss_kwargs", False) else None
            loss = soft_target_loss(outputs.logits, p, n)
            return (loss, outputs) if return_outputs else loss

    return SoftTargetTrainer


def select_trainer(target, trainer_cls):
    """``trainer_cls`` itself for ``hard`` (the default path is untouched), else
    ``soft_target_trainer(trainer_cls)``."""
    return trainer_cls if target == "hard" else soft_target_trainer(trainer_cls)


def load_split_frames(label_type, data_dir=".", target="hard", vote_weights=None):
    """``(train_df, val_df)`` with columns ``image`` (absolute crop path) and ``label`` (``LABEL2ID``), or None if
    ``<data_dir>/<label_type>/split.csv`` does not exist. For ``target`` ``soft``/``weighted`` the train frame also has
    ``soft_target`` (see ``soft_targets``; ``vote_weights`` is the DataFrame for ``weighted``)."""
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
    if target != "hard":
        train = df[df["split"] == "train"].copy()
        train["soft_target"] = soft_targets(train, target, vote_weights, path)
        return (train[cols + ["soft_target"]].reset_index(drop=True),
                df[df["split"] == "val"][cols].reset_index(drop=True))
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


def file_sha256(path):
    """Hex sha256 of a file's bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def target_settings(args):
    """``{}`` for the default hard target (so its run key and provenance are unchanged), else the target and, for
    ``weighted``, the vote-weights path and hash."""
    target = getattr(args, "target", "hard")
    if target == "hard":
        return {}
    out = {"target": target}
    if target == "weighted":
        out["vote_weights"] = os.path.abspath(args.vote_weights)
        out["vote_weights_sha256"] = file_sha256(args.vote_weights)
    return out


def default_output_dir(data_dir, label_type, target="hard"):
    """``<data_dir>/<label_type>/dinov2`` for the hard target (where evaluate.py and publish_to_hf.py look), else
    ``dinov2-<target>``, so an experiment never lands in the production model's directory."""
    return os.path.join(data_dir, label_type, "dinov2" if target == "hard" else f"dinov2-{target}")


def run_key(label_type, data_dir, args):
    """What a resumed run must share with the run that wrote the checkpoints: the split file's hash and the
    settings that shape training. ``--num-workers`` and ``--allow-cpu`` may differ between attempts. A non-default
    ``--target`` (and the vote-weights hash) is part of the key; the default adds nothing, so keys written before
    ``--target`` existed still match."""
    path = os.path.join(data_dir, label_type, "split.csv")
    split_sha256 = None
    if os.path.exists(path):
        with open(path, "rb") as f:
            split_sha256 = hashlib.sha256(f.read()).hexdigest()
    return {"label_type": label_type, "split_sha256": split_sha256, "seed": args.seed, "epochs": args.epochs,
            "best_metric": args.best_metric, "base_checkpoint": CHECKPOINT,
            **{k: v for k, v in target_settings(args).items() if k != "vote_weights"}}


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
    parser.add_argument('--output-dir', default=None,
                        help='default <data-dir>/<label_type>/dinov2, or dinov2-<target> for a non-hard --target')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--best-metric', choices=['macro_f1', 'loss'], default='macro_f1',
                        help='validation metric that picks the kept checkpoint (loss favours calibrated scores)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='dataloader worker processes (default: allocated CPUs - 1, at most 8; 0 unless the start method is fork); '
                             'with 0, decoding and augmentation starve the GPU')
    parser.add_argument('--allow-cpu', action='store_true',
                        help='run without CUDA (smoke tests only; a real run on the CPU would take weeks)')
    parser.add_argument('--target', choices=['hard', 'soft', 'weighted'], default='hard',
                        help='training target: hard 0/1 label_class (default), soft = smoothed human agreement rate '
                             'from split.csv, weighted = the same on reliability-weighted votes (--vote-weights)')
    parser.add_argument('--vote-weights', default=None,
                        help="validator_reliability.py's vote_weights.csv (required by, and only for, "
                             "--target weighted)")
    args = parser.parse_args(argv)
    if (args.target == 'weighted') != (args.vote_weights is not None):
        parser.error('--vote-weights is required by, and only used with, --target weighted')

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
    output_dir = args.output_dir or default_output_dir(args.data_dir, label_type, args.target)
    if args.num_workers is None:
        args.num_workers = default_num_workers()
    key = run_key(label_type, args.data_dir, args)
    if key["split_sha256"] is None:
        warnings.warn("no split.csv: resuming from an existing checkpoint cannot check that the data is the same.")
    resume = resume_checkpoint(output_dir, key)  # before any slow setup

    if args.target == "hard":
        frames = load_split_frames(label_type, args.data_dir)
    else:
        weights = pd.read_csv(args.vote_weights, dtype={"label_uid": str}) if args.vote_weights else None
        frames = load_split_frames(label_type, args.data_dir, args.target, weights)
        if frames is None:
            raise SystemExit(f"--target {args.target} needs {os.path.join(args.data_dir, label_type, 'split.csv')}")
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

    if args.target != "hard":
        def collate_fn(examples):  # noqa: F811 -- the soft-target batch adds soft_targets to the hard one
            batch = {"pixel_values": torch.stack([example["pixel_values"] for example in examples]),
                     "labels": torch.tensor([example["label"] for example in examples])}
            if "soft_target" in examples[0]:  # train rows only; val batches stay hard
                batch["soft_targets"] = torch.tensor([example["soft_target"] for example in examples],
                                                     dtype=torch.float32)
            return batch

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

    trainer = select_trainer(args.target, Trainer)(
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
           "split_source": "split.csv" if frames is not None else "imagefolder", **target_settings(args),
           "versions": versions}
    with open(os.path.join(output_dir, "training_run.json"), "w") as f:
        json.dump(run, f, indent=2)
        f.write("\n")
    print(f"Best checkpoint: {best}; provenance in {output_dir}/training_run.json")


if __name__ == "__main__":
    main()
