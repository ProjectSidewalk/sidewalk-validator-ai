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
    generated ``README.md`` card that records the overall content hash (sha256 of ``SHA256SUMS``). The absolute
    pano-store paths (``pano_path``, ``pano_store``) and the store marker text are dropped from the published copies.

**Binding checks** (refuse): every manifest row's ``label_type`` equals ``--label_type``; the calibration report
names the same label type; ``split.csv`` still has the sha256 recorded in ``dinov2/run_key.json`` at training time;
``predictions.csv`` covers exactly the split's evaluation rows; the dataset's content hash differs from every other
staged label type's. A train-size cross-check against ``trainer_state.json`` and out-of-order file times only warn.

**Card overrides.** Facts the pipeline does not record (per-type failure modes, the split seed, the dataset
license, DOIs) go in ``<out>/<label_type>/card_overrides.json`` (keys in ``hf_cards.OVERRIDE_KEYS``); anything still
unknown is a ``TODO:`` in the card. To hand-edit a card, stage, edit ``publish/*/README.md``, then ``--push-staged``,
which uploads the staged folders without restaging after re-verifying them against ``publish/staging.json``.

**Audit** (``--audit``; public API, no token, no local files): lists the five model and five dataset repos and
reports stray training files and datasets whose crops overlap (by path or sha256) by more than 5%.

**Upload** (``--push`` restages first, ``--push-staged`` does not; needs ``huggingface_hub>=0.34.4`` and a token from
``HF_TOKEN`` or ``huggingface-cli login`` for ``--yes``). Prints what would be uploaded, with sizes, the remote head
and the remote files that would be deleted, and how many of the staged crops already sit in another label type's
dataset repo. Only with ``--yes`` does it upload: each repo gets exactly one ``create_commit`` holding every addition
and deletion, based on the head the preview listed (``parent_commit``), so a concurrent change makes it fail rather
than interleave, and an interrupted push leaves the repo as it was. ``.gitattributes`` is kept and history is not
rewritten. ``--yes`` is refused while a card has ``TODO:`` (unless ``--allow-todo``), when a remote cannot be
listed, or when more than 5% of the crops are found in another type's dataset repo. Repos are never created under
the default prefix; under another ``--repo-prefix`` (e.g. a personal namespace for a test upload) ``--create-repo``
creates them private. ``--prune-remote`` deletes only the stray training files from existing model repos, one
commit each, again only with ``--yes``.

Exit codes: 0 success, 1 refused (``PublishError``), 3 the Hub could not be reached or listed.

Usage::

    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator          # stage only
    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator --push   # dry run
    python publish_to_hf.py --label_type CurbRamp --out /gscratch/makelab/validator --push --yes
    python publish_to_hf.py --label_type CurbRamp --out ROOT --push-staged --yes             # after editing a card
    python publish_to_hf.py --label_type CurbRamp --out ROOT --push --yes --create-repo \\
        --repo-prefix jonfroehlich/test-validator
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

PRIVATE_COLUMNS = ("pano_path",)
"""Manifest columns dropped from the published copies (absolute paths on the lab's storage)."""
PRIVATE_RUN_KEYS = ("pano_store", "pano_store_marker")
"""``generation_run.json`` keys dropped from the published copy (a lab path, and free text from the store's marker)."""

EVAL_SPLITS = ("val", "test", "test_heldout")
"""Splits evaluate.py writes predictions for."""

PREFIX_RE = re.compile(r"^[\w.-]+/[\w.-]*\w$")
"""``--repo-prefix`` must be ``<namespace>/<name>`` ending in a word character (no trailing ``-``)."""

OVERLAP_LIMIT = 0.05
"""Share of staged crops (by sha256) allowed to appear in another label type's dataset repo before upload stops."""

STAGING_RECORD = "staging.json"
"""Written to ``<T>/publish/`` (not uploaded): hashes of what was staged, for ``--push-staged``."""

EXIT_ERROR, EXIT_REMOTE = 1, 3

META_FILES = (".gitattributes", "README.md")
"""Remote files the audit ignores when comparing file lists."""


class PublishError(Exception):
    """A condition that must stop staging or upload (wrong dataset, forbidden file, missing input)."""


class RemoteError(PublishError):
    """The Hub could not be reached or listed (exit code ``EXIT_REMOTE``)."""


# ---------------------------------------------------------------------------------------------------------------------
# Repo names and forbidden files
# ---------------------------------------------------------------------------------------------------------------------

def validate_prefix(prefix):
    """Raise unless ``prefix`` looks like ``<namespace>/<name>`` (``PREFIX_RE``), so a typo cannot publish to
    ``...-ai--curbramp`` or into the token owner's own namespace.

    Example::

        >>> validate_prefix("projectsidewalk/sidewalk-validator-ai")
        >>> validate_prefix("projectsidewalk/sidewalk-validator-ai-")
        Traceback (most recent call last):
        ...
        publish_to_hf.PublishError: ...
    """
    if not PREFIX_RE.match(prefix or ""):
        raise PublishError(f"--repo-prefix {prefix!r} must look like <namespace>/<name> and end in a letter or digit")


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
    """Empty ``path`` (a staging folder) before restaging. Only ever deletes a real (not symlinked) directory whose
    parent is ``<root>/<T>/publish/`` for some label type; removing a hard link leaves the original crop in place."""
    parent = os.path.dirname(os.path.abspath(path))
    allowed = {os.path.realpath(os.path.join(root, t, "publish")) for t in LABEL_TYPES}
    for p in (path, parent):
        if os.path.islink(p):
            raise PublishError(f"refusing to stage into {path}: {p} is a symlink")
    if os.path.realpath(parent) not in allowed:
        raise PublishError(f"refusing to clear {path}: not a <out>/<label_type>/publish/ staging folder")
    if os.path.isdir(path):
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
    """This repo's HEAD commit, for the cards' lineage line, with ``-dirty`` if tracked files have uncommitted
    changes; None outside a git checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return (sha + "-dirty" if dirty else sha) or None


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


def build_croissant(summary, label_type, repo_id, split_sha256, columns, license_id=None, date_published=None):
    """A hand-built Croissant 1.1 description of a staged dataset: ``split.csv`` as the ``labels`` record set and
    the crops as the ``crops`` record set (image, split and class read from the path).

    Args:
        summary: ``dataset_summary`` output.
        split_sha256: sha256 of the published ``split.csv``.
        columns: columns of the published ``split.csv``; only the ``CROISSANT_FIELDS`` among them are described.
        date_published: ISO date for ``datePublished`` (the generation run's start date, so restaging the same data
            gives the same file); omitted when None. ``version`` is ``1.0.0+<content hash prefix>``.
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
        "version": f"1.0.0+{summary['content_hash'][:12]}",
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
    if date_published:
        doc["datePublished"] = date_published
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


def check_train_size(trainer_state, n_train, grad_accum=hf_cards.GRAD_ACCUM):
    """Warning-only cross-check: does the ``train`` split fit the optimizer steps per epoch the model recorded?

    Images per epoch ≈ steps per epoch × ``train_batch_size`` × ``grad_accum``, ± one step. It is weak (two label
    types within ±32 training images pass, and under multi-GPU training ``train_batch_size`` is per device), so it
    only warns; the binding checks (``check_bindings``) are what refuse. It would still have flagged the CurbRamp
    model (103 steps × 32 ≈ 3,300 images) against the Crosswalk crops (1,108).

    Returns:
        ``(ok, message)``; ``ok`` is None when ``trainer_state`` lacks the fields, so the check could not run.
    """
    try:
        steps = trainer_state["max_steps"] / trainer_state["num_train_epochs"]
        batch = trainer_state["train_batch_size"] * grad_accum
    except (KeyError, TypeError, ZeroDivisionError):
        return None, "train-size check skipped: trainer_state.json has no max_steps/num_train_epochs/train_batch_size"
    msg = (f"{steps:g} optimizer steps per epoch × {batch} = about {steps * batch:,.0f} training images; "
           f"the train split has {n_train:,}")
    return (steps - 1) * batch <= n_train <= (steps + 1) * batch, msg


def default_map_path(root, label_type):
    """Where ``calibration/calibrate.py`` writes the map in the README's and evaluate.py's commands."""
    return os.path.join(root, label_type, "calibration", label_type.lower() + ".json")


def check_bindings(root, label_type, best_dir, map_path, report):
    """Refuse weights, map and metrics that do not belong together; return warnings.

    Refusals (``PublishError``):

    * the calibration report next to the map names another label type (calibrate.py records ``label_type``);
    * ``sha256(<T>/split.csv)`` differs from ``split_sha256`` in ``dinov2/run_key.json``, i.e. the split changed
      after training (fine_tune_classifier.py writes the key), or the key is missing;
    * the ``label_uid`` set of ``predictions.csv`` differs from the split's ``val``/``test``/``test_heldout`` rows,
      i.e. the metrics and map were computed on another split (checked when ``metrics.json`` or a map exists).

    Warning: file times out of order (weights after predictions, or predictions after metrics/map), the sign of a
    retrain or re-evaluation whose later steps were not re-run.
    """
    base = os.path.join(root, label_type)
    if report is not None and report.get("label_type") not in (None, label_type):
        raise PublishError(f"{os.path.join(os.path.dirname(map_path), 'calibration_report.json')} is for "
                           f"{report.get('label_type')!r}, not {label_type!r}; pass the right --calibration-map")
    split_path = os.path.join(base, "split.csv")
    key = _read_json(os.path.join(base, "dinov2", "run_key.json"))
    if key is None or not key.get("split_sha256"):
        raise PublishError(f"{os.path.join(base, 'dinov2', 'run_key.json')} is missing or has no split_sha256, so "
                           f"the weights cannot be tied to {split_path}")
    if sha256_file(split_path) != key["split_sha256"]:
        raise PublishError(f"{split_path} changed after training (sha256 differs from run_key.json); retrain or "
                           f"restore the split the model was trained on")
    metrics_path, pred_path = os.path.join(base, "metrics.json"), os.path.join(base, "predictions.csv")
    has_eval = os.path.exists(metrics_path) or os.path.exists(map_path)
    if has_eval:
        if not os.path.exists(pred_path):
            raise PublishError(f"missing {pred_path}: metrics.json and the map cannot be tied to the split")
        split = pd.read_csv(split_path, usecols=["label_uid", "split"])
        want = set(split.loc[split["split"].isin(EVAL_SPLITS), "label_uid"].astype(str))
        got = set(pd.read_csv(pred_path, usecols=["label_uid"])["label_uid"].astype(str))
        if want != got:
            raise PublishError(f"{pred_path} covers {len(got)} labels but the split has {len(want)} evaluation rows "
                               f"({len(got - want)} extra, {len(want - got)} missing); re-run evaluate.py and "
                               f"calibrate.py")
    warnings = []
    weights = os.path.join(best_dir, "model.safetensors")
    if os.path.exists(pred_path) and os.path.exists(weights) and os.path.getmtime(weights) > os.path.getmtime(pred_path):
        warnings.append(f"{weights} is newer than {pred_path}: were the predictions made with these weights?")
    for later in (metrics_path, map_path):
        if os.path.exists(pred_path) and os.path.exists(later) and os.path.getmtime(pred_path) > os.path.getmtime(later):
            warnings.append(f"{pred_path} is newer than {later}: re-run the step that writes it")
    return warnings


def stage_model(root, label_type, out_dir, map_path=None, link_mode="hardlink"):
    """Stage the inference files into ``out_dir``; return a dict of what the model card needs.

    Raises ``PublishError`` for a missing weight/config file, a map that fails ``validate_api_map``, or a failed
    ``check_bindings``.
    """
    from calibration._isotonic_map import validate_api_map
    best_dir, latest_dir = resolve_checkpoints(root, label_type)
    base = os.path.join(root, label_type)
    map_path = map_path or default_map_path(root, label_type)
    api_map = _read_json(map_path)
    report = _read_json(os.path.join(os.path.dirname(map_path), "calibration_report.json")) if api_map else None
    if api_map is not None and report is None:
        raise PublishError(f"{map_path} has no calibration_report.json beside it, so its label type cannot be checked")
    for w in check_bindings(root, label_type, best_dir, map_path, report):
        print(f"WARNING: {w}")

    _reset_dir(out_dir, root)
    for name in MODEL_FILES:
        src = os.path.join(best_dir, name)
        if not os.path.exists(src):
            raise PublishError(f"missing {src}")
        _copy_in(src, os.path.join(out_dir, name), link_mode)

    state = _read_json(os.path.join(latest_dir, "trainer_state.json"), required=True)
    trainer = trim_trainer_state(state)
    _write_json(os.path.join(out_dir, "trainer_state.json"), trainer)

    metrics = _read_json(os.path.join(base, "metrics.json"))
    if metrics is not None:
        _write_json(os.path.join(out_dir, "metrics.json"), metrics)
    else:
        print(f"WARNING: no {os.path.join(base, 'metrics.json')}; the card says TODO")

    if api_map is not None:
        try:
            validate_api_map(api_map)
        except ValueError as e:
            raise PublishError(f"{map_path} is not a valid served map: {e}") from e
        with open(os.path.join(out_dir, "confidence_precision_map.json"), "w") as f:
            json.dump(api_map, f, indent=4)   # calibrate.py's own formatting
            f.write("\n")
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

def load_overrides(root, label_type):
    """``<root>/<label_type>/card_overrides.json`` (``hf_cards.OVERRIDE_KEYS``), or {} if absent; unknown keys are
    refused so a typo cannot silently leave a TODO in place.

    Example ``card_overrides.json``::

        {"failure_modes": ["Driveway aprons are often accepted as curb ramps."], "split_seed": 42,
         "dataset_license": "cc-by-4.0"}
    """
    path = os.path.join(root, label_type, "card_overrides.json")
    overrides = _read_json(path) or {}
    unknown = sorted(set(overrides) - set(hf_cards.OVERRIDE_KEYS))
    if unknown:
        raise PublishError(f"{path}: unknown keys {unknown}; allowed: {sorted(hf_cards.OVERRIDE_KEYS)}")
    return overrides


def _staged_record(model_dir, dataset_dir):
    """sha256 of every staged file except the crops (covered by SHA256SUMS) and the two cards (hand edits allowed)."""
    record = {}
    for name, folder in (("model", model_dir), ("dataset", dataset_dir)):
        for rel, _ in list_files(folder):
            if rel == "README.md" or (name == "dataset" and _is_crop_path(rel)):
                continue
            record[f"{name}/{rel}"] = sha256_file(os.path.join(folder, rel))
    return record


def _is_crop_path(rel):
    parts = rel.split("/")
    return len(parts) == 3 and parts[0] in SPLITS and parts[1] in CLASSES and parts[2].endswith(".webp")


def stage(root, label_type, prefix=DEFAULT_PREFIX, link_mode="hardlink", map_path=None, dataset_license=None):
    """Stage ``<root>/<label_type>/publish/{dataset,model}/`` with cards. Returns ``(model_dir, dataset_dir)``.

    Also writes ``publish/staging.json`` (hashes of everything staged except crops and cards), which
    ``--push-staged`` uses to prove nothing but the cards changed since.

    Example::

        model_dir, dataset_dir = stage("/gscratch/makelab/validator", "CurbRamp")
    """
    validate_prefix(prefix)
    pub = os.path.join(root, label_type, "publish")
    dataset_dir, model_dir = os.path.join(pub, "dataset"), os.path.join(pub, "model")
    m_repo, d_repo = model_repo_id(label_type, prefix), dataset_repo_id(label_type, prefix)
    commit = code_commit()
    overrides = load_overrides(root, label_type)
    dataset_license = dataset_license or overrides.get("dataset_license")

    summary, prior, gen = stage_dataset(root, label_type, dataset_dir, link_mode)
    model = stage_model(root, label_type, model_dir, map_path, link_mode)
    ok, msg = check_train_size(model["trainer"], summary["split_counts"].get("train", 0))
    if ok:
        print(f"train-size cross-check passed: {msg}")
    elif ok is None:
        print(f"WARNING: {msg}")
    else:
        print(f"WARNING: train-size cross-check failed: {msg}. Check that this dataset is the one the model was "
              f"trained on (the binding checks passed, so the split file is the training one).")

    split_csv = os.path.join(dataset_dir, "split.csv")
    croissant = build_croissant(summary, label_type, d_repo, sha256_file(split_csv),
                                list(pd.read_csv(split_csv, nrows=0).columns), dataset_license,
                                date_published=gen.get("started"))
    croissant_path = os.path.join(dataset_dir, "croissant.json")
    _write_json(croissant_path, croissant)
    print(validate_croissant(croissant_path)[1])

    ds_ctx = {"label_type": label_type, "dataset_repo": d_repo, "model_repo": m_repo, "dataset": summary,
              "prior": prior, "generation_run": gen, "code_commit": commit, "overrides": overrides}
    with open(os.path.join(dataset_dir, "README.md"), "w") as f:
        f.write(hf_cards.dataset_card(ds_ctx, dataset_license))
    m_ctx = {"label_type": label_type, "model_repo": m_repo, "dataset_repo": d_repo, "dataset": summary,
             "generation_run": gen, "code_commit": commit, "overrides": overrides, **{k: model[k] for k in (
                 "trainer", "metrics", "api_map", "calibration_report", "training_run")}}
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(hf_cards.model_card(m_ctx))

    for folder in (model_dir, dataset_dir):
        assert_publishable(folder)
    _write_json(os.path.join(pub, STAGING_RECORD),
                {"label_type": label_type, "prefix": prefix, "content_hash": summary["content_hash"],
                 "best_checkpoint": os.path.basename(model["best_dir"]),
                 "files": _staged_record(model_dir, dataset_dir)})
    return model_dir, dataset_dir


def verify_staged(root, label_type, prefix=DEFAULT_PREFIX, map_path=None):
    """Check an existing ``publish/`` before ``--push-staged`` uploads it without restaging.

    Re-runs ``assert_publishable`` and ``check_bindings``; refuses if any staged file other than the two cards
    differs from ``staging.json``, if the crops differ from ``SHA256SUMS`` or its hash from the recorded content
    hash, if the dataset card does not state that hash, or if the best checkpoint's weights changed since staging
    (a retrain). Returns ``(model_dir, dataset_dir)``.
    """
    pub = os.path.join(root, label_type, "publish")
    dataset_dir, model_dir = os.path.join(pub, "dataset"), os.path.join(pub, "model")
    record = _read_json(os.path.join(pub, STAGING_RECORD))
    if record is None:
        raise PublishError(f"no {os.path.join(pub, STAGING_RECORD)}; stage first (run without --push-staged)")
    if record.get("label_type") != label_type or record.get("prefix") != prefix:
        raise PublishError(f"{pub} was staged for {record.get('label_type')} under {record.get('prefix')!r}, not "
                           f"{label_type} under {prefix!r}; restage")
    for folder in (model_dir, dataset_dir):
        assert_publishable(folder)
    current = _staged_record(model_dir, dataset_dir)
    changed = sorted(k for k in set(current) | set(record["files"]) if current.get(k) != record["files"].get(k))
    if changed:
        raise PublishError(f"staged files changed since staging (only the README.md cards may be edited): "
                           f"{', '.join(changed[:10])}; restage")
    sums = read_sha256sums(dataset_dir)
    crops = {rel for rel, _ in list_files(dataset_dir) if _is_crop_path(rel)}
    if set(sums) != crops:
        raise PublishError(f"{dataset_dir}: crops on disk differ from SHA256SUMS; restage")
    bad = [rel for rel in sorted(crops) if sha256_file(os.path.join(dataset_dir, rel)) != sums[rel]]
    if bad:
        raise PublishError(f"{len(bad)} staged crops changed since staging (first: {bad[0]}); restage")
    content_hash = sha256_file(os.path.join(dataset_dir, "SHA256SUMS"))
    if content_hash != record["content_hash"]:
        raise PublishError("SHA256SUMS changed since staging; restage")
    with open(os.path.join(dataset_dir, "README.md")) as f:
        if content_hash not in f.read():
            raise PublishError(f"{dataset_dir}/README.md no longer states the content hash {content_hash}")
    best_dir, _ = resolve_checkpoints(root, label_type)
    if os.path.basename(best_dir) != record["best_checkpoint"] or \
            sha256_file(os.path.join(best_dir, "model.safetensors")) != record["files"]["model/model.safetensors"]:
        raise PublishError("the best checkpoint changed since staging (a retrain?); restage")
    map_path = map_path or default_map_path(root, label_type)
    report = _read_json(os.path.join(os.path.dirname(map_path), "calibration_report.json"))
    for w in check_bindings(root, label_type, best_dir, map_path, report):
        print(f"WARNING: {w}")
    return model_dir, dataset_dir


# ---------------------------------------------------------------------------------------------------------------------
# Remote listings (public API with requests; authenticated with huggingface_hub)
# ---------------------------------------------------------------------------------------------------------------------

def fetch_repo_listing(repo_type, repo_id, get=None):
    """A public repo's head as ``{"sha", "files": [{"path", "size", "blob", "sha256"}]}`` from the Hub API, or None
    if it does not exist or is not public. ``sha256`` is the LFS content hash (None for non-LFS files). ``get``
    defaults to ``requests.get`` (injectable for tests). Network failures raise ``RemoteError``."""
    import requests
    get = get or requests.get
    kind = "models" if repo_type == "model" else "datasets"
    try:
        r = get(f"{HF_API}/{kind}/{repo_id}", params={"blobs": "true"}, timeout=60)
        if r.status_code in (401, 403, 404):
            return None
        r.raise_for_status()
        body = r.json()
    except requests.RequestException as e:
        raise RemoteError(f"could not list {repo_type} {repo_id}: {e}") from e
    return {"sha": body.get("sha"),
            "files": [{"path": s["rfilename"], "size": s.get("size"), "blob": s.get("blobId"),
                       "sha256": (s.get("lfs") or {}).get("sha256")} for s in body.get("siblings", [])]}


def fetch_repo_files(repo_type, repo_id, get=None):
    """Just the ``files`` of ``fetch_repo_listing`` (None if the repo is missing or private)."""
    listing = fetch_repo_listing(repo_type, repo_id, get)
    return None if listing is None else listing["files"]


def _lfs_sha256(repo_file):
    """The LFS sha256 of a ``huggingface_hub.RepoFile`` (attribute or dict form, across versions), or None."""
    lfs = getattr(repo_file, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)


def authenticated_listing(api, repo_type, repo_id):
    """Like ``fetch_repo_listing`` but through an authenticated ``HfApi`` (sees private and gated repos): the head
    sha from ``repo_info`` and the files of exactly that revision from ``list_repo_tree``."""
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # pragma: no cover - only reached with a fake api in tests
        RepositoryNotFoundError = LookupError
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
        files = [{"path": f.path, "size": f.size, "blob": getattr(f, "blob_id", None), "sha256": _lfs_sha256(f)}
                 for f in api.list_repo_tree(repo_id, repo_type=repo_type, revision=info.sha, recursive=True)
                 if not hasattr(f, "tree_id")]   # RepoFolder entries have a tree_id, files do not
    except RepositoryNotFoundError:
        return None
    except Exception as e:  # HTTP and connection errors from huggingface_hub/requests
        raise RemoteError(f"could not list {repo_type} {repo_id}: {e}") from e
    return {"sha": info.sha, "files": files}


def crop_hashes(files):
    """The LFS sha256 of every crop (``<split>/<class>/<name>.webp``) in a listing; crops without one are skipped."""
    return {f["sha256"] for f in files or [] if f.get("sha256") and _is_crop_path(f["path"])}


def overlap_fraction(a, b):
    """``|a ∩ b| / min(|a|, |b|)``: 1.0 when one set is contained in the other, 0.0 when either is empty."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def remote_overlaps(staged_hashes, label_type, prefix, lister):
    """Share of the staged crops (by sha256) found in each OTHER label type's dataset repo.

    Returns ``[(repo_id, fraction or None if the repo is missing, n_remote_crops_with_hash)]``. Catches a dataset
    built from another type's crops even when its file names, manifest and local checks all look right."""
    out = []
    for other in LABEL_TYPES:
        if other == label_type:
            continue
        repo = dataset_repo_id(other, prefix)
        listing = lister("dataset", repo)
        if listing is None:
            out.append((repo, None, 0))
            continue
        remote = crop_hashes(listing["files"])
        out.append((repo, len(staged_hashes & remote) / len(staged_hashes) if staged_hashes else 0.0, len(remote)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Audit (public API, requests only)
# ---------------------------------------------------------------------------------------------------------------------

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
    ``split_counts`` ({split/class: n}), ``card_size``. ``duplicates``: ``(repo_a, repo_b, name_overlap,
    content_overlap)`` for every pair of datasets whose crop paths or crop sha256s overlap by more than
    ``OVERLAP_LIMIT`` (``overlap_fraction``; ``content_overlap`` is None when the listings carry no sha256).
    """
    models, datasets, crops = [], [], {}
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
            crops[repo] = ({f["path"] for f in files if _is_crop_path(f["path"])}, crop_hashes(files))

    duplicates = []
    repos = list(crops)
    for i, a in enumerate(repos):
        for b in repos[i + 1:]:
            names = overlap_fraction(crops[a][0], crops[b][0])
            content = overlap_fraction(crops[a][1], crops[b][1]) if crops[a][1] and crops[b][1] else None
            if names > OVERLAP_LIMIT or (content or 0) > OVERLAP_LIMIT:
                duplicates.append((a, b, names, content))
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
    lines += ["", "Dataset repos:", "", "| repo | files | card | crops by split/class | overlaps with |",
              "|---|---|---|---|---|"]
    for r in result["datasets"]:
        if not r["exists"]:
            lines.append(f"| {r['repo']} | missing or private | | | |")
            continue
        twins = [(b if a == r["repo"] else a, n, c) for a, b, n, c in result["duplicates"] if r["repo"] in (a, b)]
        twin_text = ", ".join(f"{t} ({n:.0%} of names, {'n/a' if c is None else f'{c:.0%}'} of content)"
                              for t, n, c in twins)
        counts = ", ".join(f"{k} {v}" for k, v in r["split_counts"].items())
        lines.append(f"| {r['repo']} | {r['n_files']} | {_card_text(r['card_size'])} | {counts} | {twin_text} |")
    n_stray = sum(len(r["stray"]) for r in result["models"])
    lines += ["", f"{n_stray} stray training files in model repos; {len(result['duplicates'])} pair(s) of datasets "
                  f"overlapping by more than {OVERLAP_LIMIT:.0%} (of the smaller set)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# Upload (huggingface_hub, optional)
# ---------------------------------------------------------------------------------------------------------------------

def _hf_api(required=True):
    """An authenticated ``HfApi``, or None when not ``required`` and there is no library or token."""
    try:
        import huggingface_hub
    except ImportError as e:
        if not required:
            return None
        raise PublishError("--push --yes and --prune-remote --yes need huggingface_hub>=0.34.4: "
                           "pip install huggingface_hub") from e
    token = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    if not token:
        if not required:
            return None
        raise PublishError("no Hugging Face token: set HF_TOKEN or run `huggingface-cli login`")
    return huggingface_hub.HfApi(token=token)


def _operation_classes():
    """``(CommitOperationAdd, CommitOperationDelete)``; a function so tests can substitute stand-ins."""
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    except ImportError as e:
        raise PublishError("--push --yes needs huggingface_hub>=0.34.4") from e
    return CommitOperationAdd, CommitOperationDelete


def count_todos(folder):
    """Number of ``TODO:`` markers in ``folder``'s ``README.md``."""
    path = os.path.join(folder, "README.md")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return f.read().count(hf_cards.TODO)


def read_sha256sums(folder):
    """``{relpath: sha256}`` from ``folder/SHA256SUMS``."""
    out = {}
    with open(os.path.join(folder, "SHA256SUMS")) as f:
        for line in f:
            digest, _, rel = line.rstrip("\n").partition("  ")
            out[rel] = digest
    return out


def plan_push(folder, repo_id, repo_type, lister):
    """Everything a push of ``folder`` to ``repo_id`` would do, without doing it.

    Returns a dict: ``folder``, ``repo_id``, ``repo_type``, ``files`` ([(relpath, size)]), ``todos``, ``listing``
    (the remote head, None if the repo does not exist), ``listing_error`` (str or None) and ``deletes`` (remote
    paths not in the folder, ``.gitattributes`` excepted).
    """
    assert_publishable(folder)
    plan = {"folder": folder, "repo_id": repo_id, "repo_type": repo_type, "files": list_files(folder),
            "todos": count_todos(folder), "listing": None, "listing_error": None, "deletes": []}
    try:
        plan["listing"] = lister(repo_type, repo_id)
    except RemoteError as e:
        plan["listing_error"] = str(e)
    if plan["listing"] is not None:
        local = {p for p, _ in plan["files"]}
        plan["deletes"] = sorted(f["path"] for f in plan["listing"]["files"]
                                 if f["path"] not in local and f["path"] != ".gitattributes")
    return plan


def print_plan(plan, yes):
    files = plan["files"]
    print(f"\n{'UPLOAD' if yes else 'DRY RUN'}: {plan['repo_type']} {plan['repo_id']} <- {plan['folder']} "
          f"({len(files)} files, {_human_size(sum(s for _, s in files))}; {plan['todos']} TODO markers in the card)")
    for p, s in files if len(files) <= 40 else files[:20]:
        print(f"  {_human_size(s):>10}  {p}")
    if len(files) > 40:
        print(f"  ... and {len(files) - 20} more")
    if plan["listing_error"]:
        print(f"  remote: could not list ({plan['listing_error']})")
    elif plan["listing"] is None:
        print("  remote: the repo does not exist (or is not visible without a token)")
    else:
        gone = plan["deletes"]
        print(f"  remote head {plan['listing']['sha']}; files this commit deletes: {len(gone)}"
              + (": " + ", ".join(gone[:10]) + (" ..." if len(gone) > 10 else "") if gone else ""))


def check_plan(plan, prefix, allow_todo, create_repo):
    """Raise ``PublishError`` if ``plan`` must not be committed (checked for every repo before any upload)."""
    if plan["todos"] and not allow_todo:
        raise PublishError(f"{plan['folder']}/README.md still has {plan['todos']} '{hf_cards.TODO}' markers; fill "
                           f"them (card_overrides.json, or edit the card and use --push-staged) or pass --allow-todo")
    if plan["listing_error"]:
        raise PublishError(f"refusing to upload to {plan['repo_id']}: {plan['listing_error']}")
    if plan["listing"] is None:
        if prefix == DEFAULT_PREFIX:
            raise PublishError(f"{plan['repo_id']} does not exist; repos under the default prefix are never created "
                               f"by this script")
        if not create_repo:
            raise PublishError(f"{plan['repo_id']} does not exist; pass --create-repo to create it (private)")


def commit_push(api, plan, message, private=True):
    """Upload one planned push as exactly ONE commit (``HfApi.create_commit``), based on the remote head the plan
    listed (``parent_commit``), so a concurrent change to the repo makes the commit fail instead of interleaving.

    ``upload_folder`` is avoided on purpose: with ``hf_xet`` installed, huggingface_hub 2.x streams large folders
    as many "(part N)" commits, and an interrupted push would leave old and new crops mixed at HEAD. Returns the
    commit info."""
    listing = plan["listing"]
    if listing is None:
        api.create_repo(plan["repo_id"], repo_type=plan["repo_type"], private=private, exist_ok=False)
        listing = authenticated_listing(api, plan["repo_type"], plan["repo_id"])
        if listing is None:
            raise RemoteError(f"created {plan['repo_id']} but cannot list it")
        plan = dict(plan, deletes=[f["path"] for f in listing["files"]
                                   if f["path"] not in {p for p, _ in plan["files"]} and f["path"] != ".gitattributes"])
    add, delete = _operation_classes()
    operations = ([delete(path_in_repo=p) for p in plan["deletes"]]
                  + [add(path_in_repo=p, path_or_fileobj=os.path.join(plan["folder"], p)) for p, _ in plan["files"]])
    info = api.create_commit(repo_id=plan["repo_id"], repo_type=plan["repo_type"], operations=operations,
                             commit_message=message, parent_commit=listing["sha"])
    print(f"  committed {plan['repo_id']}: {getattr(info, 'commit_url', info)}")
    return info


def push(label_type, model_dir, dataset_dir, prefix=DEFAULT_PREFIX, what="both", yes=False, allow_todo=False,
         create_repo=False, api=None, public_lister=None):
    """Plan, check and (with ``yes``) commit the staged folders. All checks for every repo run before any upload.

    The preview uses the authenticated listing when a token is available (it sees private repos), else the public
    API. The dataset's crops are compared by sha256 with the other four label types' dataset repos; more than
    ``OVERLAP_LIMIT`` of them found in one is reported, and refused with ``yes``.
    """
    validate_prefix(prefix)
    if api is None:
        api = _hf_api(required=yes)
    lister = ((lambda rt, rid: authenticated_listing(api, rt, rid)) if api is not None
              else public_lister or fetch_repo_listing)
    commit = code_commit()
    suffix = f" (sidewalk-validator-ai {commit[:12]})" if commit else ""
    plans = []
    if what in ("both", "dataset"):
        plans.append((plan_push(dataset_dir, dataset_repo_id(label_type, prefix), "dataset", lister),
                      f"Publish {label_type} dataset with card and SHA256SUMS{suffix}"))
    if what in ("both", "model"):
        plans.append((plan_push(model_dir, model_repo_id(label_type, prefix), "model", lister),
                      f"Publish {label_type} model: inference files and card{suffix}"))
    for plan, _ in plans:
        print_plan(plan, yes)

    problems = []
    if what in ("both", "dataset"):
        staged = set(read_sha256sums(dataset_dir).values())
        try:
            overlaps = remote_overlaps(staged, label_type, prefix, lister)
        except RemoteError as e:
            overlaps = None
            problems.append(f"could not compare with the other dataset repos: {e}")
        for repo, frac, n in overlaps or []:
            text = "missing" if frac is None else f"{frac:.1%} of the staged crops ({n} hashed crops there)"
            print(f"  overlap with {repo}: {text}")
            if frac is not None and frac > OVERLAP_LIMIT:
                problems.append(f"{frac:.1%} of the staged {label_type} crops are already in {repo}")
    if not yes:
        for p in problems:
            print(f"  WARNING: {p}")
        print("\n(dry run; add --yes to upload)")
        return plans
    for plan, _ in plans:
        check_plan(plan, prefix, allow_todo, create_repo)
    if problems:
        raise PublishError("; ".join(problems))
    for plan, message in plans:
        commit_push(api, plan, message, private=prefix != DEFAULT_PREFIX)
    return plans


def prune_remote(repo_id, yes=False, fetch=fetch_repo_listing, api=None):
    """Delete the training-only files from an existing model repo in one commit (``delete_files`` is a single
    ``create_commit``, based on the listed head; no history rewrite); dry run unless ``yes``. Returns the paths."""
    listing = fetch("model", repo_id)
    if listing is None:
        print(f"{repo_id}: missing or private, nothing to prune")
        return []
    files = listing["files"]
    stray = [f["path"] for f in files if forbidden_reason(f["path"])]
    size = sum(f["size"] or 0 for f in files if f["path"] in stray)
    if not stray:
        print(f"{repo_id}: no stray training files")
        return []
    print(f"{repo_id}: {'deleting' if yes else 'would delete'} {', '.join(stray)} ({_human_size(size)})")
    if yes:
        api = api or _hf_api()
        api.delete_files(repo_id=repo_id, delete_patterns=stray, repo_type="model", parent_commit=listing["sha"],
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
                   help=f"repo names are <prefix>-<type> and <prefix>-dataset-<type> (default {DEFAULT_PREFIX}); "
                        "must be <namespace>/<name>")
    p.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink",
                   help="how crops and weights enter the staging folder (hardlink falls back to copy across "
                        "filesystems); never moved")
    p.add_argument("--calibration-map", help="calibrated map JSON (default <out>/<type>/calibration/<type>.json)")
    p.add_argument("--dataset-license", help="Hub license id for the dataset card (or dataset_license in "
                                             "card_overrides.json; default: a TODO, issue #2)")
    p.add_argument("--audit", action="store_true", help="report on the live repos (public API, no token)")
    p.add_argument("--push", action="store_true", help="stage, then upload (dry run unless --yes)")
    p.add_argument("--push-staged", action="store_true",
                   help="upload the existing publish/ without restaging (keeps hand edits to the README.md cards), "
                        "after re-checking it (dry run unless --yes)")
    p.add_argument("--prune-remote", action="store_true",
                   help="delete stray training files from existing model repos (dry run unless --yes)")
    p.add_argument("--what", choices=("both", "model", "dataset"), default="both", help="which repo(s) to upload")
    p.add_argument("--create-repo", action="store_true",
                   help="create missing repos (private), only for a non-default --repo-prefix")
    p.add_argument("--yes", action="store_true", help="really upload / delete")
    p.add_argument("--allow-todo", action="store_true", help="upload cards that still contain TODO: markers")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate_prefix(args.repo_prefix)
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
        if args.push and args.push_staged:
            raise PublishError("use --push (restage, then upload) or --push-staged (upload what is staged), not both")
        if args.push_staged:
            model_dir, dataset_dir = verify_staged(args.out, args.label_type, args.repo_prefix, args.calibration_map)
            print(f"verified {model_dir} and {dataset_dir}")
        else:
            model_dir, dataset_dir = stage(args.out, args.label_type, args.repo_prefix, args.link_mode,
                                           args.calibration_map, args.dataset_license)
            print(f"staged {model_dir} and {dataset_dir}")
        for name, folder in (("model", model_dir), ("dataset", dataset_dir)):
            n = count_todos(folder)
            print(f"  {name} card: {n} TODO markers" + (" to fill before --yes" if n else ""))
        if args.push or args.push_staged:
            push(args.label_type, model_dir, dataset_dir, args.repo_prefix, args.what, args.yes, args.allow_todo,
                 args.create_repo)
        return 0
    except RemoteError as e:
        print(f"ERROR (Hub unreachable or refused): {e}", file=sys.stderr)
        return EXIT_REMOTE
    except PublishError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
