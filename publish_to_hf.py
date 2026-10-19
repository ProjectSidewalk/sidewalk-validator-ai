"""Stage, audit and upload one label type's validator model and dataset to Hugging Face, with generated cards.

Replaces the manual "upload the checkpoint folder" step that put optimizer state in every model repo and a copy of
the Crosswalk crops in the CurbRamp dataset repo (https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7).
Cards: https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2.

**Staging** (the default; no network, no token). Given the pipeline output root (``--out``, the same root every other
script writes to), builds two folders under ``<out>/<label_type>/publish/``:

``model/``
    ``model.safetensors``, ``config.json`` and ``preprocessor_config.json`` from the best checkpoint (found the way
    ``evaluate.py`` finds it: ``best_model_checkpoint`` in the latest complete checkpoint's ``trainer_state.json``);
    that ``trainer_state.json`` trimmed to its log history and best-checkpoint fields; the calibrated map from
    ``calibration/calibrate.py`` renamed ``confidence_precision_map.json``, with its ``calibration_report.json``;
    ``metrics.json``; ``training_run.json``; and a generated ``README.md`` card. Training-only files
    (``optimizer.pt``, ``scheduler.pt``, ``rng_state*.pth``, ``training_args.bin``, ``scaler.pt``, any
    ``checkpoint-*`` directory) are refused: staging copies from an allow-list, and every folder is checked again
    before upload.
``dataset/``
    Crops as ``<split>/<class>/<city>_<label_id>.webp``, driven by ``split.csv`` (never a directory listing), hard
    linked or copied (``--link-mode``), never moved; ``split.csv`` (``crop_path`` rewritten to the repo layout),
    ``manifest.csv``, this label type's ``deployment_priors.json`` entry, ``generation_run.json``,
    ``dataset_info.json``, ``SHA256SUMS`` (sha256 of every crop, sorted by path), ``croissant.json`` and a
    generated ``README.md`` card that records the overall content hash (sha256 of ``SHA256SUMS``). The manifest's
    ``label_type`` column must equal ``--label_type``, and a dataset whose content hash equals another label type's
    staged dataset is refused (the #7 bug). The absolute pano-store paths (``pano_path``, ``pano_store``) are dropped
    from the published copies.

**Consistency check.** The model's ``trainer_state.json`` implies how many training images it saw per epoch
(optimizer steps per epoch × batch 8 × accumulation 4); staging stops if the dataset's ``train`` split does not
match (``--skip-train-size-check`` to override). This is the check that exposed the CurbRamp/Crosswalk swap.

**Audit** (``--audit``; public API, no token, no local files): lists the five model and five dataset repos and
reports stray training files and datasets with identical file lists.

**Upload** (``--push``, needs ``huggingface_hub`` and a token from ``HF_TOKEN`` or ``huggingface-cli login``):
stages, then prints exactly what would be uploaded, with sizes, and what remote files would be deleted. Only with
``--yes`` does it upload, as one commit per repo that makes the repo mirror the staged folder (stale remote files are
deleted; ``.gitattributes`` is kept; history is not rewritten). Cards that still contain ``TODO:`` are refused unless
``--allow-todo``. ``--prune-remote`` deletes only the stray training files from existing model repos, one commit
each, again only with ``--yes``. ``--repo-prefix`` publishes elsewhere, e.g. a personal namespace for a test upload.

Usage::

    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator          # stage only
    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator --push   # dry run
    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator --push --yes
    python publish_to_hf.py --label_type CurbRamp --out ROOT --push --yes --repo-prefix jonfroehlich/test-validator
    python publish_to_hf.py --audit
    python publish_to_hf.py --prune-remote --label_type all            # dry run: what would be deleted
    python publish_to_hf.py --prune-remote --label_type all --yes
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date

import pandas as pd

import hf_cards

LABEL_TYPES = ("CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "Crosswalk")
CLASSES = ("correct", "incorrect")
SPLITS = ("train", "val", "test", "test_heldout")
DEFAULT_PREFIX = "projectsidewalk/sidewalk-validator-ai"
HF_API = "https://huggingface.co/api"

MODEL_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
"""Copied from the best checkpoint as they are."""

FORBIDDEN_NAMES = ("optimizer.pt", "scheduler.pt", "training_args.bin", "scaler.pt")
FORBIDDEN_GLOBS = ("rng_state*.pth",)
FORBIDDEN_DIR_GLOBS = ("checkpoint-*",)

TRAINER_STATE_KEEP = ("best_global_step", "best_metric", "best_model_checkpoint", "epoch", "global_step",
                      "max_steps", "num_train_epochs", "train_batch_size", "logging_steps", "log_history")
"""``trainer_state.json`` fields kept in the published copy; the rest (callback state, flops, trial fields) is
training-loop bookkeeping."""

GRAD_ACCUM = 4
"""``gradient_accumulation_steps`` in fine_tune_classifier.py; with ``train_batch_size`` it gives images per step."""

PRIVATE_COLUMNS = ("pano_path",)
"""Manifest columns dropped from the published copies (absolute paths on the lab's storage)."""
PRIVATE_RUN_KEYS = ("pano_store",)

META_FILES = (".gitattributes", "README.md")
"""Remote files the audit ignores when comparing file lists."""


class PublishError(Exception):
    """A condition that must stop staging or upload (wrong dataset, forbidden file, missing input)."""


# ---------------------------------------------------------------------------------------------------------------------
# Repo names and forbidden files
# ---------------------------------------------------------------------------------------------------------------------

def model_repo_id(label_type, prefix=DEFAULT_PREFIX):
    """``projectsidewalk/sidewalk-validator-ai-curbramp`` for ``CurbRamp``."""
    return f"{prefix}-{label_type.lower()}"


def dataset_repo_id(label_type, prefix=DEFAULT_PREFIX):
    """``projectsidewalk/sidewalk-validator-ai-dataset-curbramp`` for ``CurbRamp``."""
    return f"{prefix}-dataset-{label_type.lower()}"


def forbidden_reason(relpath):
    """Why ``relpath`` (a path inside a repo) must never be published, or None if it may be.

    Example::

        >>> forbidden_reason("optimizer.pt")
        'training-only file optimizer.pt'
        >>> forbidden_reason("checkpoint-2060/model.safetensors")
        'inside checkpoint directory checkpoint-2060'
        >>> forbidden_reason("model.safetensors") is None
        True
    """
    parts = relpath.replace("\\", "/").split("/")
    for part in parts[:-1]:
        if any(fnmatch.fnmatch(part, g) for g in FORBIDDEN_DIR_GLOBS):
            return f"inside checkpoint directory {part}"
    name = parts[-1]
    if any(fnmatch.fnmatch(name, g) for g in FORBIDDEN_DIR_GLOBS):
        return f"checkpoint directory {name}"
    if name in FORBIDDEN_NAMES or any(fnmatch.fnmatch(name, g) for g in FORBIDDEN_GLOBS):
        return f"training-only file {name}"
    return None


def list_files(folder):
    """Every file under ``folder`` as ``(relpath, size)``, sorted by relpath (``/`` separators)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in filenames:
            full = os.path.join(dirpath, name)
            out.append((os.path.relpath(full, folder).replace(os.sep, "/"), os.path.getsize(full)))
    return sorted(out)


def assert_publishable(folder):
    """Raise ``PublishError`` if ``folder`` contains anything ``forbidden_reason`` rejects (checked before upload,
    so a training file dropped into a staged folder by hand is still caught)."""
    bad = [(p, forbidden_reason(p)) for p, _ in list_files(folder) if forbidden_reason(p)]
    if bad:
        raise PublishError(f"{folder} contains files that must not be published: "
                           + "; ".join(f"{p} ({why})" for p, why in bad))


def _copy_in(src, dst, link_mode):
    """Hard link (falling back to a copy across filesystems) or copy ``src`` to ``dst``; never moves."""
    if forbidden_reason(os.path.basename(src)):
        raise PublishError(f"refusing to stage {src}: {forbidden_reason(os.path.basename(src))}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # another filesystem, or links not allowed: copy instead
    shutil.copy2(src, dst)


def _reset_dir(path, root):
    """Empty ``path`` (a staging folder) before restaging. Only ever deletes inside ``<root>/<T>/publish/``; removing
    a hard link leaves the original crop in place."""
    if os.path.isdir(path):
        if os.sep + "publish" + os.sep not in os.path.abspath(path) + os.sep:
            raise PublishError(f"refusing to clear {path}: not a publish/ staging folder")
        shutil.rmtree(path)
    os.makedirs(path)


def _read_json(path, required=False):
    if not os.path.exists(path):
        if required:
            raise PublishError(f"missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
        f.write("\n")


def sha256_file(path, chunk=1 << 20):
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def code_commit():
    """This repo's HEAD commit, for the cards' lineage line; None outside a git checkout."""
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Dataset staging
# ---------------------------------------------------------------------------------------------------------------------

def check_label_type(df, label_type, what):
    """Raise unless every row of ``df`` has ``label_type == label_type`` (catches a dataset copied from another type)."""
    if "label_type" not in df.columns:
        raise PublishError(f"{what} has no label_type column")
    found = sorted(df["label_type"].dropna().astype(str).unique())
    if found != [label_type]:
        raise PublishError(f"{what} has label_type {found}, expected only {label_type!r}; refusing to publish it "
                           f"as the {label_type} dataset")


def check_label_uids(df, what):
    """``label_uid`` must be unique and equal ``<city>:<label_id>`` (label_id alone repeats across cities)."""
    if not df["label_uid"].is_unique:
        raise PublishError(f"{what}: label_uid is not unique")
    expected = df["city"].astype(str) + ":" + df["label_id"].astype(str)
    bad = df["label_uid"].astype(str) != expected
    if bad.any():
        raise PublishError(f"{what}: {int(bad.sum())} rows whose label_uid is not <city>:<label_id> "
                           f"(first: {df.loc[bad, 'label_uid'].iloc[0]})")


def write_sha256sums(folder, relpaths):
    """Write ``SHA256SUMS`` (``<sha256>  <relpath>`` per crop, sorted by relpath) and return the content hash, the
    sha256 of that file's bytes. Same crops at the same paths give the same hash on any machine.

    Example::

        content_hash = write_sha256sums("CurbRamp/publish/dataset", ["test/correct/teaneck_1.webp", ...])
        # verify a download: sha256sum -c SHA256SUMS && sha256sum SHA256SUMS
    """
    lines = [f"{sha256_file(os.path.join(folder, p))}  {p}\n" for p in sorted(relpaths)]
    text = "".join(lines).encode()
    with open(os.path.join(folder, "SHA256SUMS"), "wb") as f:
        f.write(text)
    return hashlib.sha256(text).hexdigest()


def dataset_summary(split_df, content_hash):
    """The counts the dataset card and ``dataset_info.json`` report, computed from the published ``split.csv``.

    Returns a dict: ``n``, ``splits`` (present, in canonical order), ``counts`` ({split: {class: n}}),
    ``split_counts``, ``class_totals``, ``cities`` ({city: n}, sorted), ``heldout_cities``, ``capture_years``
    ({min, max} from the leading year of ``image_capture_date``), ``n_panoramas`` and ``content_hash``.
    """
    splits = [s for s in SPLITS if (split_df["split"] == s).any()]
    counts = {s: {c: int(((split_df["split"] == s) & (split_df["label_class"] == c)).sum()) for c in CLASSES}
              for s in splits}
    years = []
    if "image_capture_date" in split_df.columns:
        for v in split_df["image_capture_date"].dropna().astype(str):
            m = re.match(r"^(\d{4})", v)
            if m:
                years.append(int(m.group(1)))
    return {"n": int(len(split_df)), "splits": splits, "counts": counts,
            "split_counts": {s: sum(counts[s].values()) for s in splits},
            "class_totals": {c: int((split_df["label_class"] == c).sum()) for c in CLASSES},
            "cities": {k: int(v) for k, v in sorted(split_df["city"].astype(str).value_counts().items())},
            "heldout_cities": sorted(split_df.loc[split_df["split"] == "test_heldout", "city"].astype(str).unique()),
            "capture_years": {"min": min(years) if years else None, "max": max(years) if years else None},
            "n_panoramas": int(split_df["pano_id"].nunique()) if "pano_id" in split_df.columns else None,
            "content_hash": content_hash}


def check_not_duplicate(root, label_type, content_hash):
    """Refuse a dataset whose content hash equals another label type's staged one under the same root."""
    for other in LABEL_TYPES:
        if other == label_type:
            continue
        info = _read_json(os.path.join(root, other, "publish", "dataset", "dataset_info.json"))
        if info and info.get("content_hash") == content_hash:
            raise PublishError(f"the staged {label_type} dataset has the same content hash as the staged {other} "
                               f"dataset ({content_hash[:12]}); one of them is a copy of the other")


def stage_dataset(root, label_type, out_dir, link_mode="hardlink"):
    """Stage ``<root>/<label_type>``'s crops and metadata into ``out_dir``; return ``(summary, prior, generation_run)``.

    ``out_dir`` is emptied first. Crops are placed by ``split.csv`` only; a row whose crop is missing stops staging.
    """
    base = os.path.join(root, label_type)
    split_path, manifest_path = os.path.join(base, "split.csv"), os.path.join(base, "manifest.csv")
    for p in (split_path, manifest_path):
        if not os.path.exists(p):
            raise PublishError(f"missing {p} (run data_generation.py and split_data.py first)")
    split = pd.read_csv(split_path)
    manifest = pd.read_csv(manifest_path)
    check_label_type(manifest, label_type, manifest_path)
    check_label_type(split, label_type, split_path)
    check_label_uids(split, split_path)
    unknown = sorted(set(split["split"].astype(str)) - set(SPLITS))
    if unknown:
        raise PublishError(f"{split_path} has unknown split values {unknown}")
    bad_class = sorted(set(split["label_class"].astype(str)) - set(CLASSES))
    if bad_class:
        raise PublishError(f"{split_path} has unknown label_class values {bad_class}")

    _reset_dir(out_dir, root)
    rel_paths, missing = [], []
    for row in split.itertuples(index=False):
        src = row.crop_path if os.path.isabs(row.crop_path) else os.path.join(base, row.crop_path)
        if not os.path.exists(src):
            missing.append(src)
            continue
        if not src.lower().endswith(".webp"):
            raise PublishError(f"{src}: expected a .webp crop")
        rel = f"{row.split}/{row.label_class}/{row.city}_{row.label_id}.webp"
        _copy_in(src, os.path.join(out_dir, rel), link_mode)
        rel_paths.append(rel)
    if missing:
        raise PublishError(f"{len(missing)} crops listed in {split_path} do not exist (first: {missing[0]})")

    published = split.drop(columns=[c for c in PRIVATE_COLUMNS if c in split.columns])
    published["crop_path"] = rel_paths
    published.to_csv(os.path.join(out_dir, "split.csv"), index=False)
    manifest.drop(columns=[c for c in PRIVATE_COLUMNS if c in manifest.columns]).to_csv(
        os.path.join(out_dir, "manifest.csv"), index=False)

    priors = _read_json(os.path.join(root, "deployment_priors.json")) or {}
    prior = priors.get(label_type)
    if prior is None:
        print(f"WARNING: no {label_type} entry in {os.path.join(root, 'deployment_priors.json')}; the card says TODO")
    else:
        _write_json(os.path.join(out_dir, "deployment_priors.json"), {label_type: prior})
    gen = _read_json(os.path.join(base, "generation_run.json")) or {}
    if gen:
        _write_json(os.path.join(out_dir, "generation_run.json"),
                    {k: v for k, v in gen.items() if k not in PRIVATE_RUN_KEYS})

    content_hash = write_sha256sums(out_dir, rel_paths)
    check_not_duplicate(root, label_type, content_hash)
    summary = dataset_summary(published, content_hash)
    _write_json(os.path.join(out_dir, "dataset_info.json"),
                {"label_type": label_type, **summary, "selection_sha256": gen.get("selection_sha256")})
    return summary, prior, gen


# ---------------------------------------------------------------------------------------------------------------------
# Croissant
# ---------------------------------------------------------------------------------------------------------------------

CROISSANT_CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "arrayShape": "cr:arrayShape", "citeAs": "cr:citeAs",
    "column": "cr:column", "conformsTo": "dct:conformsTo", "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/", "description": {"@container": "@language"},
    "equivalentProperty": "cr:equivalentProperty", "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes", "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5",
    "name": {"@container": "@language"}, "parentField": "cr:parentField", "path": "cr:path",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
    "replace": "cr:replace", "samplingRate": "cr:samplingRate", "sc": "https://schema.org/",
    "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform",
}
"""The standard Croissant 1.1 ``@context`` (as ``mlcroissant`` 1.1 expects it)."""

CROISSANT_FIELDS = {
    "label_uid": ("sc:Text", "Unique key, <city>:<label_id>."),
    "city": ("sc:Text", "Project Sidewalk deployment (city_id)."),
    "label_id": ("sc:Integer", "Label id within its city; NOT unique across cities."),
    "pano_id": ("sc:Text", "Google Street View panorama id."),
    "label_class": ("sc:Text", "correct or incorrect, from human validation votes."),
    "human_agree": ("sc:Integer", "Human Agree votes."),
    "human_disagree": ("sc:Integer", "Human Disagree votes."),
    "image_capture_date": ("sc:Text", "Capture date of the panorama."),
    "depth_m": ("sc:Float", "Depth Anything V2 metric depth at the label, metres."),
    "hard": ("sc:Integer", "1 if the label is in the hard-example list."),
    "split": ("sc:Text", "train, val, test or test_heldout."),
    "crop_path": ("sc:Text", "Path of the crop within this repository."),
}


def build_croissant(summary, label_type, repo_id, split_sha256, columns, license_id=None):
    """A hand-built Croissant 1.1 description of a staged dataset: ``split.csv`` as the ``labels`` record set and
    the crops as the ``crops`` record set (image, split and class read from the path).

    Args:
        summary: ``dataset_summary`` output.
        split_sha256: sha256 of the published ``split.csv``.
        columns: columns of the published ``split.csv``; only the ``CROISSANT_FIELDS`` among them are described.
    """
    base_url = f"https://huggingface.co/datasets/{repo_id}"
    label_fields = [{"@type": "cr:Field", "@id": f"labels/{c}", "name": c, "description": desc, "dataType": dtype,
                     "source": {"fileObject": {"@id": "split-csv"}, "extract": {"column": c}}}
                    for c, (dtype, desc) in CROISSANT_FIELDS.items() if c in columns]

    def from_path(field_id, name, desc, dtype, regex=None, prop="fullpath"):
        src = {"fileSet": {"@id": "crop-files"}, "extract": {"fileProperty": prop}}
        if regex:
            src["transform"] = {"regex": regex}
        return {"@type": "cr:Field", "@id": f"crops/{field_id}", "name": name, "description": desc,
                "dataType": dtype, "source": src}

    doc = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": repo_id.split("/")[-1],
        "description": (f"Label-centred street-level crops of Project Sidewalk {label_type} labels, marked correct or "
                        f"incorrect from human validation votes. {summary['n']} crops; content hash "
                        f"{summary['content_hash']}."),
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "url": base_url,
        "version": "1.0.0",
        "datePublished": date.today().isoformat(),   # the staging date; the upload follows it
        "citeAs": hf_cards.CITATION_BIBTEX,
        "creator": {"@type": "sc:Organization", "name": "Project Sidewalk, Makeability Lab, University of Washington",
                    "url": "https://projectsidewalk.org"},
        "keywords": ["accessibility", "sidewalks", "street view", "crowdsourcing", label_type],
        "distribution": [
            {"@type": "cr:FileObject", "@id": "repo", "name": "repo", "description": "The Hugging Face repository.",
             "contentUrl": base_url, "encodingFormat": "git+https", "sha256": "main"},
            {"@type": "cr:FileObject", "@id": "split-csv", "name": "split.csv",
             "description": "One row per crop.", "contentUrl": f"{base_url}/resolve/main/split.csv",
             "encodingFormat": "text/csv", "sha256": split_sha256},
            {"@type": "cr:FileSet", "@id": "crop-files", "name": "crop-files", "description": "The WebP crops.",
             "containedIn": {"@id": "repo"}, "encodingFormat": "image/webp", "includes": "*/*/*.webp"},
        ],
        "recordSet": [
            {"@type": "cr:RecordSet", "@id": "labels", "name": "labels", "key": {"@id": "labels/label_uid"},
             "description": "split.csv: one record per crop.", "field": label_fields},
            {"@type": "cr:RecordSet", "@id": "crops", "name": "crops", "description": "The crops, by path.",
             "field": [from_path("image", "image", "The crop.", "sc:ImageObject", prop="content"),
                       from_path("path", "path", "Path within the repository.", "sc:Text"),
                       from_path("split", "split", "Split, from the first path component.", "sc:Text",
                                 regex=r"^(?:.*/)?([^/]+)/[^/]+/[^/]+\.webp$"),
                       from_path("label_class", "label_class", "Class, from the second path component.",
                                 "sc:Text", regex=r"^(?:.*/)?([^/]+)/[^/]+\.webp$")]},
        ],
    }
    if license_id:
        doc["license"] = license_id
    return doc


def validate_croissant(path):
    """Validate ``path`` with ``mlcroissant`` if it is installed. Returns ``(validated, message)``."""
    try:
        import mlcroissant as mlc
    except ImportError:
        return False, "mlcroissant not installed; croissant.json written but not validated"
    try:
        mlc.Dataset(jsonld=path)
    except Exception as e:  # mlcroissant raises its own ValidationError and others
        raise PublishError(f"{path} does not validate: {e}") from e
    return True, "croissant.json validated with mlcroissant"


# ---------------------------------------------------------------------------------------------------------------------
# Model staging
# ---------------------------------------------------------------------------------------------------------------------

def resolve_checkpoints(root, label_type):
    """``(best_dir, latest_dir)`` under ``<root>/<label_type>/dinov2/``.

    The best checkpoint is the one ``evaluate.find_best_checkpoint`` names (``best_model_checkpoint`` in the latest
    complete checkpoint's ``trainer_state.json``). That value is the path as the Trainer saw it during training
    (e.g. ``CurbRamp/dinov2/checkpoint-2060``), so it is resolved by its basename under this ``dinov2/``.
    """
    import evaluate
    from fine_tune_classifier import is_complete_checkpoint
    base = os.path.join(root, label_type, "dinov2")
    if not os.path.isdir(base):
        raise PublishError(f"missing {base} (run fine_tune_classifier.py first)")
    try:
        best = evaluate.find_best_checkpoint(label_type, root)
    except (FileNotFoundError, ValueError) as e:
        raise PublishError(str(e)) from e
    best_dir = os.path.join(base, os.path.basename(os.path.normpath(best)))
    if not is_complete_checkpoint(best_dir):
        raise PublishError(f"best_model_checkpoint {best!r} is not a complete checkpoint under {base}")
    latest = max((d for d in os.listdir(base) if is_complete_checkpoint(os.path.join(base, d))),
                 key=lambda d: int(d.split("-")[1]))
    return best_dir, os.path.join(base, latest)


def trim_trainer_state(state):
    """Keep ``TRAINER_STATE_KEEP``; ``best_model_checkpoint`` becomes its basename (the training-time path is not
    meaningful to a reader)."""
    out = {k: state[k] for k in TRAINER_STATE_KEEP if k in state}
    if out.get("best_model_checkpoint"):
        out["best_model_checkpoint"] = os.path.basename(os.path.normpath(out["best_model_checkpoint"]))
    return out


def check_train_size(trainer_state, n_train, grad_accum=GRAD_ACCUM):
    """Raise if the number of ``train`` crops does not fit the optimizer steps per epoch the model recorded.

    Images per epoch ≈ steps per epoch × ``train_batch_size`` × ``grad_accum``; one step either way is allowed for
    rounding. Returns a one-line description, or None when ``trainer_state`` lacks the fields.

    Example: the published CurbRamp model has 103 steps per epoch × 32 ≈ 3,300 images, while the published
    "CurbRamp" dataset has 1,108 train crops (the Crosswalk set), so this raises.
    """
    try:
        steps = trainer_state["max_steps"] / trainer_state["num_train_epochs"]
        batch = trainer_state["train_batch_size"] * grad_accum
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    lo, hi = (steps - 1) * batch, (steps + 1) * batch
    msg = f"{steps:g} optimizer steps per epoch × {batch} = about {steps * batch:,.0f} training images; train split has {n_train:,}"
    if not lo <= n_train <= hi:
        raise PublishError(f"model and dataset disagree: {msg}. The dataset is not the one this model was trained "
                           f"on (pass --skip-train-size-check only if you know why)")
    return msg


def default_map_path(root, label_type):
    """Where ``calibration/calibrate.py`` writes the map in the README's and evaluate.py's commands."""
    return os.path.join(root, label_type, "calibration", label_type.lower() + ".json")


def stage_model(root, label_type, out_dir, map_path=None, link_mode="hardlink"):
    """Stage the inference files into ``out_dir``; return a dict of what the model card needs.

    Raises ``PublishError`` for a missing weight/config file or a map that fails ``validate_api_map``.
    """
    from calibration._isotonic_map import validate_api_map
    best_dir, latest_dir = resolve_checkpoints(root, label_type)
    _reset_dir(out_dir, root)
    for name in MODEL_FILES:
        src = os.path.join(best_dir, name)
        if not os.path.exists(src):
            raise PublishError(f"missing {src}")
        _copy_in(src, os.path.join(out_dir, name), link_mode)

    state = _read_json(os.path.join(latest_dir, "trainer_state.json"), required=True)
    trainer = trim_trainer_state(state)
    _write_json(os.path.join(out_dir, "trainer_state.json"), trainer)

    base = os.path.join(root, label_type)
    metrics = _read_json(os.path.join(base, "metrics.json"))
    if metrics is not None:
        _write_json(os.path.join(out_dir, "metrics.json"), metrics)
    else:
        print(f"WARNING: no {os.path.join(base, 'metrics.json')}; the card says TODO")

    map_path = map_path or default_map_path(root, label_type)
    api_map = _read_json(map_path)
    report = None
    if api_map is not None:
        try:
            validate_api_map(api_map)
        except ValueError as e:
            raise PublishError(f"{map_path} is not a valid served map: {e}") from e
        with open(os.path.join(out_dir, "confidence_precision_map.json"), "w") as f:
            json.dump(api_map, f, indent=4)   # calibrate.py's own formatting
            f.write("\n")
        report = _read_json(os.path.join(os.path.dirname(map_path), "calibration_report.json"))
        if report is not None:
            _write_json(os.path.join(out_dir, "calibration_report.json"), report)
    else:
        print(f"WARNING: no calibrated map at {map_path}; the card says TODO (pass --calibration-map)")

    run = _read_json(os.path.join(base, "dinov2", "training_run.json")) or {}
    if run:
        _write_json(os.path.join(out_dir, "training_run.json"), run)
    return {"best_dir": best_dir, "trainer": trainer, "metrics": metrics, "api_map": api_map,
            "calibration_report": report, "training_run": run}


# ---------------------------------------------------------------------------------------------------------------------
# Staging both, with cards
# ---------------------------------------------------------------------------------------------------------------------

def stage(root, label_type, prefix=DEFAULT_PREFIX, link_mode="hardlink", map_path=None, dataset_license=None,
          check_train=True):
    """Stage ``<root>/<label_type>/publish/{dataset,model}/`` with cards. Returns ``(model_dir, dataset_dir)``.

    Example::

        model_dir, dataset_dir = stage("/gscratch/makelab/validator", "CurbRamp")
    """
    pub = os.path.join(root, label_type, "publish")
    dataset_dir, model_dir = os.path.join(pub, "dataset"), os.path.join(pub, "model")
    m_repo, d_repo = model_repo_id(label_type, prefix), dataset_repo_id(label_type, prefix)
    commit = code_commit()

    summary, prior, gen = stage_dataset(root, label_type, dataset_dir, link_mode)
    model = stage_model(root, label_type, model_dir, map_path, link_mode)
    if check_train:
        msg = check_train_size(model["trainer"], summary["split_counts"].get("train", 0))
        if msg:
            print(f"train-size check passed: {msg}")

    split_csv = os.path.join(dataset_dir, "split.csv")
    croissant = build_croissant(summary, label_type, d_repo, sha256_file(split_csv),
                                list(pd.read_csv(split_csv, nrows=0).columns), dataset_license)
    croissant_path = os.path.join(dataset_dir, "croissant.json")
    _write_json(croissant_path, croissant)
    print(validate_croissant(croissant_path)[1])

    ds_ctx = {"label_type": label_type, "dataset_repo": d_repo, "model_repo": m_repo, "dataset": summary,
              "prior": prior, "generation_run": gen, "code_commit": commit}
    with open(os.path.join(dataset_dir, "README.md"), "w") as f:
        f.write(hf_cards.dataset_card(ds_ctx, dataset_license))
    m_ctx = {"label_type": label_type, "model_repo": m_repo, "dataset_repo": d_repo, "dataset": summary,
             "generation_run": gen, "code_commit": commit, **{k: model[k] for k in (
                 "trainer", "metrics", "api_map", "calibration_report", "training_run")}}
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(hf_cards.model_card(m_ctx))

    for folder in (model_dir, dataset_dir):
        assert_publishable(folder)
    return model_dir, dataset_dir


# ---------------------------------------------------------------------------------------------------------------------
# Audit (public API, requests only)
# ---------------------------------------------------------------------------------------------------------------------

def fetch_repo_files(repo_type, repo_id, get=None):
    """Files of a public repo as ``[{"path", "size", "blob"}]`` from the Hub API, or None if it does not exist or
    is not public. ``get`` defaults to ``requests.get`` (injectable for tests)."""
    if get is None:
        import requests
        get = requests.get
    kind = "models" if repo_type == "model" else "datasets"
    r = get(f"{HF_API}/{kind}/{repo_id}", params={"blobs": "true"}, timeout=60)
    if r.status_code in (401, 403, 404):
        return None
    r.raise_for_status()
    return [{"path": s["rfilename"], "size": s.get("size"), "blob": s.get("blobId")}
            for s in r.json().get("siblings", [])]


def _human_size(n):
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


STUB_CARD_BYTES = 500
"""A README.md smaller than this is reported as a stub (the live model repos carry a 24-byte ``license: mit`` one)."""


def _card_size(files):
    """Size of the repo's README.md, or None if it has none."""
    return next((f["size"] for f in files or [] if f["path"] == "README.md"), None)


def _card_text(size):
    if size is None:
        return "no"
    return f"stub ({_human_size(size)})" if size < STUB_CARD_BYTES else "yes"


def audit(prefix=DEFAULT_PREFIX, fetch=fetch_repo_files):
    """Inspect the five model and five dataset repos. Returns ``{"models": [...], "datasets": [...],
    "duplicates": [...]}``; ``format_audit`` renders it.

    Each model row: ``repo``, ``exists``, ``n_files``, ``stray`` (``[(path, size)]`` of training-only files),
    ``card_size`` (README.md bytes or None). Each dataset row: ``repo``, ``exists``, ``n_files``,
    ``split_counts`` ({split/class: n}), ``card_size``. ``duplicates``: ``(repo_a, repo_b, same_content)`` for every pair of datasets with identical file
    lists (ignoring ``.gitattributes``/``README.md``); ``same_content`` is True when the blob ids match too.
    """
    models, datasets, listings = [], [], {}
    for t in LABEL_TYPES:
        repo = model_repo_id(t, prefix)
        files = fetch("model", repo)
        row = {"repo": repo, "exists": files is not None, "n_files": len(files or []), "stray": [],
               "card_size": _card_size(files)}
        for f in files or []:
            if forbidden_reason(f["path"]):
                row["stray"].append((f["path"], f["size"]))
        models.append(row)

        repo = dataset_repo_id(t, prefix)
        files = fetch("dataset", repo)
        counts = {}
        for f in files or []:
            parts = f["path"].split("/")
            if len(parts) == 3 and parts[0] in SPLITS + ("validation",):
                key = f"{parts[0]}/{parts[1]}"
                counts[key] = counts.get(key, 0) + 1
        datasets.append({"repo": repo, "exists": files is not None, "n_files": len(files or []),
                         "split_counts": dict(sorted(counts.items())), "card_size": _card_size(files)})
        if files:
            listings[repo] = {f["path"]: f["blob"] for f in files if f["path"] not in META_FILES}

    duplicates = []
    repos = list(listings)
    for i, a in enumerate(repos):
        for b in repos[i + 1:]:
            if listings[a] and set(listings[a]) == set(listings[b]):
                duplicates.append((a, b, listings[a] == listings[b]))
    return {"models": models, "datasets": datasets, "duplicates": duplicates}


def format_audit(result):
    """Markdown tables for ``audit()``'s result."""
    lines = ["Model repos:", "", "| repo | files | card | stray training files |", "|---|---|---|---|"]
    for r in result["models"]:
        if not r["exists"]:
            lines.append(f"| {r['repo']} | missing or private | | |")
            continue
        stray = ", ".join(f"{p} ({_human_size(s)})" for p, s in r["stray"]) or "none"
        lines.append(f"| {r['repo']} | {r['n_files']} | {_card_text(r['card_size'])} | {stray} |")
    lines += ["", "Dataset repos:", "", "| repo | files | card | crops by split/class | identical to |",
              "|---|---|---|---|---|"]
    for r in result["datasets"]:
        if not r["exists"]:
            lines.append(f"| {r['repo']} | missing or private | | | |")
            continue
        twins = [(b if a == r["repo"] else a, same) for a, b, same in result["duplicates"] if r["repo"] in (a, b)]
        twin_text = ", ".join(f"{t} ({'same content' if same else 'same names'})" for t, same in twins) or ""
        counts = ", ".join(f"{k} {v}" for k, v in r["split_counts"].items())
        lines.append(f"| {r['repo']} | {r['n_files']} | {_card_text(r['card_size'])} | {counts} | {twin_text} |")
    n_stray = sum(len(r["stray"]) for r in result["models"])
    lines += ["", f"{n_stray} stray training files in model repos; {len(result['duplicates'])} pair(s) of datasets "
                  f"with identical file lists."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Upload (huggingface_hub, optional)
# ---------------------------------------------------------------------------------------------------------------------

def _hf_api():
    """An authenticated ``HfApi``; raises ``PublishError`` without ``huggingface_hub`` or a token."""
    try:
        import huggingface_hub
    except ImportError as e:
        raise PublishError("--push and --prune-remote need huggingface_hub: pip install huggingface_hub") from e
    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    if not token:
        raise PublishError("no Hugging Face token: set HF_TOKEN or run `huggingface-cli login`")
    return huggingface_hub.HfApi(token=token)


def count_todos(folder):
    """Number of ``TODO:`` markers in ``folder``'s ``README.md``."""
    path = os.path.join(folder, "README.md")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return f.read().count(hf_cards.TODO)


def push_folder(folder, repo_id, repo_type, message, yes=False, allow_todo=False, fetch=fetch_repo_files, api=None):
    """Upload ``folder`` to ``repo_id`` as one commit that makes the repo mirror the folder; dry run unless ``yes``.

    Returns the list of ``(relpath, size)`` that was (or would be) uploaded.
    """
    assert_publishable(folder)
    todos = count_todos(folder)
    if todos and not allow_todo:
        raise PublishError(f"{folder}/README.md still has {todos} '{hf_cards.TODO}' markers; fill them or pass "
                           f"--allow-todo")
    plan = list_files(folder)
    total = sum(s for _, s in plan)
    print(f"\n{'UPLOAD' if yes else 'DRY RUN'}: {repo_type} {repo_id} <- {folder} "
          f"({len(plan)} files, {_human_size(total)}; {todos} TODO markers in the card)")
    for p, s in plan if len(plan) <= 40 else plan[:20]:
        print(f"  {_human_size(s):>10}  {p}")
    if len(plan) > 40:
        print(f"  ... and {len(plan) - 20} more")
    try:
        remote = fetch(repo_type, repo_id)
    except Exception as e:  # network trouble only affects the preview
        remote, why = None, str(e)
    else:
        why = "repo does not exist yet (or is private)"
    if remote is None:
        print(f"  remote: could not list ({why})")
    else:
        local = {p for p, _ in plan}
        gone = sorted(f["path"] for f in remote if f["path"] not in local and f["path"] != ".gitattributes")
        print(f"  remote files that this commit deletes: {len(gone)}"
              + (": " + ", ".join(gone[:10]) + (" ..." if len(gone) > 10 else "") if gone else ""))
    if not yes:
        print("  (dry run; add --yes to upload)")
        return plan
    api = api or _hf_api()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=folder, commit_message=message,
                             delete_patterns="*")   # .gitattributes is never deleted by huggingface_hub
    print(f"  committed: {getattr(info, 'commit_url', info)}")
    return plan


def prune_remote(repo_id, yes=False, fetch=fetch_repo_files, api=None):
    """Delete the training-only files from an existing model repo in one commit (no history rewrite); dry run
    unless ``yes``. Returns the list of paths (to be) deleted."""
    files = fetch("model", repo_id)
    if files is None:
        print(f"{repo_id}: missing or private, nothing to prune")
        return []
    stray = [f["path"] for f in files if forbidden_reason(f["path"])]
    size = sum(f["size"] or 0 for f in files if f["path"] in stray)
    if not stray:
        print(f"{repo_id}: no stray training files")
        return []
    print(f"{repo_id}: {'deleting' if yes else 'would delete'} {', '.join(stray)} ({_human_size(size)})")
    if yes:
        api = api or _hf_api()
        api.delete_files(repo_id=repo_id, delete_patterns=stray, repo_type="model",
                         commit_message="Remove training-only files (optimizer, scheduler, RNG state, training "
                                        "args); see ProjectSidewalk/sidewalk-validator-ai#7")
    else:
        print("  (dry run; add --yes to delete)")
    return stray


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label_type", choices=LABEL_TYPES + ("all",),
                   help="label type to stage/push; 'all' only with --prune-remote")
    p.add_argument("--out", help="pipeline output root holding <label_type>/ and deployment_priors.json")
    p.add_argument("--repo-prefix", default=DEFAULT_PREFIX,
                   help=f"repo names are <prefix>-<type> and <prefix>-dataset-<type> (default {DEFAULT_PREFIX})")
    p.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink",
                   help="how crops and weights enter the staging folder (hardlink falls back to copy across "
                        "filesystems); never moved")
    p.add_argument("--calibration-map", help="calibrated map JSON (default <out>/<type>/calibration/<type>.json)")
    p.add_argument("--dataset-license", help="Hub license id for the dataset card (default: a TODO; issue #2)")
    p.add_argument("--skip-train-size-check", action="store_true",
                   help="stage even if the train split does not match the model's trainer_state.json")
    p.add_argument("--audit", action="store_true", help="report on the live repos (public API, no token)")
    p.add_argument("--push", action="store_true", help="stage, then upload (dry run unless --yes)")
    p.add_argument("--prune-remote", action="store_true",
                   help="delete stray training files from existing model repos (dry run unless --yes)")
    p.add_argument("--what", choices=("both", "model", "dataset"), default="both", help="which repo(s) --push uploads")
    p.add_argument("--yes", action="store_true", help="really upload / delete")
    p.add_argument("--allow-todo", action="store_true", help="upload cards that still contain TODO: markers")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.audit:
            print(format_audit(audit(args.repo_prefix)))
            return 0
        if not args.label_type:
            raise PublishError("--label_type is required")
        if args.prune_remote:
            types = LABEL_TYPES if args.label_type == "all" else (args.label_type,)
            for t in types:
                prune_remote(model_repo_id(t, args.repo_prefix), yes=args.yes)
            return 0
        if args.label_type == "all":
            raise PublishError("--label_type all is only for --prune-remote; stage one label type at a time")
        if not args.out:
            raise PublishError("--out is required")
        model_dir, dataset_dir = stage(args.out, args.label_type, args.repo_prefix, args.link_mode,
                                       args.calibration_map, args.dataset_license, not args.skip_train_size_check)
        print(f"staged {model_dir} and {dataset_dir}")
        for name, folder in (("model", model_dir), ("dataset", dataset_dir)):
            n = count_todos(folder)
            if n:
                print(f"  {name} card: {n} TODO markers to fill before --push")
        if args.push:
            commit = code_commit()
            suffix = f" (sidewalk-validator-ai {commit[:10]})" if commit else ""
            if args.what in ("both", "dataset"):
                push_folder(dataset_dir, dataset_repo_id(args.label_type, args.repo_prefix), "dataset",
                            f"Publish {args.label_type} dataset with card and SHA256SUMS{suffix}", args.yes,
                            args.allow_todo)
            if args.what in ("both", "model"):
                push_folder(model_dir, model_repo_id(args.label_type, args.repo_prefix), "model",
                            f"Publish {args.label_type} model: inference files and card{suffix}", args.yes,
                            args.allow_todo)
        return 0
    except PublishError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
