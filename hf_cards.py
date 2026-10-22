"""Hugging Face model and dataset cards for the validator models, filled from the pipeline's own output files.

Used by ``publish_to_hf.py``; kept separate so the card text can be read and edited without wading through the
staging code. Both cards follow the Hugging Face templates (model card:
https://huggingface.co/docs/hub/model-cards, dataset card: https://huggingface.co/docs/hub/datasets-cards): YAML front
matter the Hub indexes, then a Markdown body.

Every number comes from a file the pipeline wrote (``metrics.json``, the calibrated map and its
``calibration_report.json``, ``split.csv``, ``manifest.csv``, ``trainer_state.json``, ``generation_run.json``,
``deployment_priors.json``). A field that cannot be filled from those files is written as ``TODO: <what is missing>``
so it is visible in review, never guessed; ``publish_to_hf.py --push`` refuses to upload a card that still has one
unless ``--allow-todo`` is given.

Example::

    from hf_cards import model_card, dataset_card
    text = model_card(ctx)        # ctx: the dict publish_to_hf.stage() builds from the staged files
"""
import json
import math
import re

import data_generation as dg

TODO = "TODO:"

TRAIN_BATCH_SIZE = 8
"""``per_device_train_batch_size`` in fine_tune_classifier.py (a test checks it against that file)."""
GRAD_ACCUM = 4
"""``gradient_accumulation_steps`` in fine_tune_classifier.py (a test checks it against that file)."""

OVERRIDE_KEYS = {
    "failure_modes": "list of strings: per-label-type failure modes for the model card",
    "split_seed": "int: the --seed split_data.py ran with (it is not recorded in split.csv)",
    "dataset_license": "Hub license id for the dataset (same as --dataset-license)",
    "doi": "the dataset's DOI, once minted",
    "model_doi": "the model's DOI, once minted",
}
"""Keys allowed in ``<out>/<label_type>/card_overrides.json``: facts the pipeline's files do not record."""
ENCODER_NAMES = {"vits": "ViT-S", "vitb": "ViT-B", "vitl": "ViT-L"}
GATE = 0.92
"""SidewalkWebpage's ``ai-validation-min-accuracy`` (the same value as ``calibration._isotonic_map.GATE``)."""

CODE_URL = "https://github.com/ProjectSidewalk/sidewalk-validator-ai"
API_URL = "https://github.com/ProjectSidewalk/sidewalk-ai-api"
PLATFORM_URL = "https://github.com/ProjectSidewalk/SidewalkWebpage"

LABEL_TYPE_DESCRIPTIONS = {
    "CurbRamp": "a curb ramp",
    "NoCurbRamp": "a missing curb ramp (a corner or crossing that should have a ramp but does not)",
    "Obstacle": "an obstacle in the pedestrian path",
    "SurfaceProblem": "a sidewalk surface problem (cracks, heaving, uneven or broken surface)",
    "Crosswalk": "a crosswalk",
}

# The README's bibtex, so the cards and the README cite the same thing.
CITATION_BIBTEX = """@inproceedings{saha2019projectsidewalk,
  author    = {Saha, Manaswi and Saugstad, Michael and Maddali, Hanuma Teja and Zeng, Aileen and Holland, Ryan and Bower, Steven and Dash, Aditya and Chen, Sage and Li, Anthony and Hara, Kotaro and Froehlich, Jon},
  title     = {Project Sidewalk: A Web-based Crowdsourcing Tool for Collecting Sidewalk Accessibility Data at Scale},
  booktitle = {Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems},
  year      = {2019},
  doi       = {10.1145/3290605.3300292}
}"""

LINEAGE = """| year | work | relation |
|---|---|---|
| 2019 | Weld et al., ASSETS'19 ([doi](https://doi.org/10.1145/3308561.3353798)) | First deep-learning auto-validation of Project Sidewalk labels on depth-sized crops. |
| 2022 | Duan et al., ASSETS'22 ([doi](https://doi.org/10.1145/3517428.3550381)) | Validated training data and cross-city transfer; the direct ancestor of this pipeline. |
| 2023 | Oquab et al., DINOv2 ([arXiv](https://arxiv.org/abs/2304.07193)) | The pretrained backbone. |
| 2024 | Yang et al., Depth Anything V2 ([code](https://github.com/DepthAnything/Depth-Anything-V2)) | Metric depth that sizes each crop. |"""


# ---------------------------------------------------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------------------------------------------------

def fixed_crop_half_px(*runs):
    """The ``fixed_crop_half_px`` any of ``runs`` (``generation_run.json`` / ``training_run.json`` dicts, top level
    or under ``settings``) records, else None. Such crops come from data_generation.py's fixed-size ablation
    (issue #21); the cards describe depth-sized crops, which is what serving makes.

    >>> fixed_crop_half_px({"tilt_beta": 0.95}, {"settings": {"fixed_crop_half_px": 300}})
    300
    >>> fixed_crop_half_px({}, None) is None
    True
    """
    for run in runs:
        run = run or {}
        for d in (run, run.get("settings") or {}):
            if d.get("fixed_crop_half_px") is not None:
                return d["fixed_crop_half_px"]
    return None


def require_depth_sized(*runs):
    """Raise ``ValueError`` if any of ``runs`` records fixed-size crops: a card would describe the wrong crops."""
    half = fixed_crop_half_px(*runs)
    if half is not None:
        raise ValueError(f"the crops were cut at a fixed half-width of {half} px (--fixed-crop-half-px, the issue #21 "
                         f"ablation), but serving sizes each crop by depth ({dg.CROP_DEPTH_CONSTANT} / depth_m): "
                         f"this is an ablation-only model or dataset, and it is not published")


_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9_./-]*$")
_YAML_WORDS = {"true", "false", "yes", "no", "on", "off", "null", "none", "~"}


def _yaml_scalar(v):
    """One YAML scalar. Strings are left bare only when that cannot change their meaning; otherwise JSON-quoted,
    which is valid YAML. NaN becomes null (YAML has .nan, but the Hub's metadata parser expects numbers or null)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _yaml_float(v)
    s = str(v)
    if _BARE.match(s) and s.lower() not in _YAML_WORDS:
        return s
    return json.dumps(s)


def _yaml_float(v):
    """A float YAML 1.1 readers also parse as a float: always a decimal point, signed exponent, no inf/nan.

    Example::

        >>> [_yaml_float(x) for x in (0.5, 1e-05, 3.0, 2e20)]
        ['0.5', '1.0e-05', '3.0', '2.0e+20']
    """
    if not math.isfinite(v):
        return "null"
    s = format(v, ".10g")
    mantissa, _, exp = s.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return mantissa + ("e" + exp if exp else "")


def _yaml_lines(obj, indent):
    pad = " " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.extend(_yaml_lines(v, indent + 2))
            elif isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}: {'{}' if isinstance(v, dict) else '[]'}")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                child = _yaml_lines(item, indent + 2)
                child[0] = f"{pad}- " + child[0].lstrip()   # first line of the item shares the dash
                lines.extend(child)
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
    return lines


def yaml_front_matter(meta, comments=()):
    """Render ``meta`` (nested dicts, lists and scalars) as a ``---`` delimited YAML block.

    A minimal emitter, so publishing does not need PyYAML. ``comments`` are emitted as ``# ...`` lines at the top
    (used for TODOs that have no valid metadata value yet).

    Example::

        >>> print(yaml_front_matter({"license": "mit", "tags": ["a", "b"]}))
        ---
        license: mit
        tags:
          - a
          - b
        ---
        <BLANKLINE>
    """
    body = [f"# {c}" for c in comments] + _yaml_lines(meta, 0)
    return "---\n" + "\n".join(body) + "\n---\n"


def fmt(x, nd=3):
    """A number for a card table (thousands separators for ints), or ``n/a`` when the source file did not have it."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def size_category(n):
    """The Hub's ``size_categories`` bucket for ``n`` examples (``"1K<n<10K"`` etc.)."""
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                        (1_000_000, "100K<n<1M")):
        if n < limit:
            return name
    return "1M<n<10M"


def macro_f1(slice_metrics):
    """Mean of the per-class F1 scores in one ``evaluate.slice_metrics`` entry, or None."""
    per_class = slice_metrics.get("per_class") or {}
    f1s = [r["f1-score"] for r in per_class.values()]
    return sum(f1s) / len(f1s) if f1s else None


def gate_thresholds(api_map, gate=GATE):
    """For each verdict, the lowest confidence the map serves at or above ``gate``, or None if it never does.

    This is read off the map alone: a verdict with confidence at or above this value becomes a real vote.

    Example::

        >>> gate_thresholds({"correct": {"0.5": 0.5, "0.9": 0.93}, "incorrect": {"0.5": 0.4}})
        {'correct': 0.9, 'incorrect': None}
    """
    out = {}
    for verdict, class_map in api_map.items():
        passing = [float(k) for k, v in class_map.items() if v >= gate]
        out[verdict] = min(passing) if passing else None
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------------------------------------------------

SLICE_NAMES = {"test": "test (cities seen in training)", "test_heldout": "test_heldout (held-out cities)",
               "hard": "hard (test + test_heldout rows the deployed model got wrong)",
               "gold": "gold (test + test_heldout labels with >= {gold_min_votes} human votes, every voter's "
                       "reliability >= {gold_min_reliability})"}


def _metrics_section(metrics):
    if not metrics:
        return f"{TODO} metrics.json not found; run evaluate.py before publishing.\n"
    metrics = {k: v for k, v in metrics.items() if k != "per_city"}  # evaluate.py --per-city: not a slice
    lines = ["| slice | n (correct / incorrect) | accuracy | balanced accuracy | AUC | macro-F1 |",
             "|---|---|---|---|---|---|"]
    # An empty gold slice (evaluate.py --vote-weights, no label qualified) says nothing in a card.
    metrics = {k: m for k, m in metrics.items() if not (k == "gold" and not m.get("n"))}
    for name, m in metrics.items():
        nb = m.get("n_by_class", {})
        label = SLICE_NAMES.get(name, name)
        if name == "gold":
            label = label.format(gold_min_votes=m.get("gold_min_votes", "?"),
                                 gold_min_reliability=m.get("gold_min_reliability", "?"))
        lines.append(f"| {label} | {fmt(m.get('n'))} ({fmt(nb.get('correct'))} / "
                     f"{fmt(nb.get('incorrect'))}) | {fmt(m.get('accuracy'))} | {fmt(m.get('balanced_accuracy'))} | "
                     f"{fmt(m.get('auc'))} | {fmt(macro_f1(m))} |")
    lines += ["", "Per class:", "", "| slice | class | precision | recall | F1 | support |", "|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        for cls, r in (m.get("per_class") or {}).items():
            lines.append(f"| {name} | {cls} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1-score'])} | "
                         f"{fmt(int(r['support']))} |")
    rap = (metrics.get("test") or {}).get("recall_at_precision") or {}
    if rap:
        lines += ["", "Recall at precision on the test split (the score threshold that achieves it in brackets):", "",
                  "| class | P ≥ 0.90 | P ≥ 0.92 | P ≥ 0.95 | P ≥ 0.99 |", "|---|---|---|---|---|"]
        for cls, table in rap.items():
            cells = []
            for level in ("0.90", "0.92", "0.95", "0.99"):
                r = table.get(level)
                if r is None:
                    cells.append("n/a")
                else:
                    th = "n/a" if r["threshold"] is None else f"{r['threshold']:.4f}"
                    cells.append(f"{r['recall']:.3f} (≥ {th})")
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _calibration_section(ctx):
    api_map, rep = ctx.get("api_map"), ctx.get("calibration_report")
    if not api_map:
        return (f"{TODO} no calibrated map found; run `python -m calibration.calibrate` (see the training repo's "
                f"README) before publishing.\n")
    th = gate_thresholds(api_map)
    lines = [f"`confidence_precision_map.json` maps the model's softmax confidence for its verdict to the estimated "
             f"probability that the verdict is right. It is fitted by `calibration/calibrate.py` with isotonic "
             f"regression, reweighted from the training class mix to the deployment class prior, and is a drop-in "
             f"for `sidewalk-ai-api/accuracy_mappings/{ctx['label_type'].lower()}.json`.", "",
             f"Lowest confidence at which the map reaches the {GATE} gate:", "",
             "| verdict | confidence at the gate |", "|---|---|"]
    for v in ("correct", "incorrect"):
        lines.append(f"| {v} | {'never reaches the gate' if th.get(v) is None else f'{th[v]:g}'} |")
    lines.append("")
    if not rep:
        lines.append(f"{TODO} calibration_report.json not found next to the map, so the deployment prior, ECE and "
                     f"coverage at the gate are not reported.")
        return "\n".join(lines) + "\n"
    prior = rep.get("prior_deploy") or {}
    src = rep.get("prior_source") or {}
    lines.append(f"Deployment prior: correct {fmt(prior.get('correct'))}, incorrect {fmt(prior.get('incorrect'))} "
                 f"(from {src.get('source', 'deployment_priors.json')}, measured {src.get('measured_on', 'n/a')}). "
                 f"Fitted on {fmt(rep.get('n_fit'))} `{rep.get('fit_split')}` rows; evaluated on "
                 f"{fmt(rep.get('n_eval'))} `{rep.get('eval_split')}` rows, reweighted to the same prior.")
    ev = rep.get("eval")
    if isinstance(ev, dict):
        lines += ["", f"Overall expected calibration error (ECE): {fmt(ev.get('overall_ece'), 4)}.", "",
                  f"| verdict | n | weighted share | precision | ECE | coverage at {GATE} | precision at {GATE} |",
                  "|---|---|---|---|---|---|---|"]
        for v in ("correct", "incorrect"):
            r = ev.get(v) or {}
            lines.append(f"| {v} | {fmt(r.get('n'))} | {fmt(r.get('weighted_share'))} | {fmt(r.get('precision'))} | "
                         f"{fmt(r.get('ece'), 4)} | {fmt(r.get('gate_coverage'))} | {fmt(r.get('gate_precision'))} |")
    else:
        lines += ["", f"Evaluation of the map: n/a ({rep.get('eval_error', 'no eval rows')})."]
    hb = rep.get("hold_back") or {}
    if hb.get("verdicts"):
        unc = hb.get("uncapped_eval") or {}
        held = ", ".join(f"`{v}`" for v in hb["verdicts"])
        lines += ["", f"**Held back:** the {held} map is capped at {fmt(hb.get('cap'), 4)}, so that verdict is served with "
                      f"its estimated accuracy but never reaches the {GATE} gate and never becomes a vote. "
                      f"Reason: {hb.get('reason', 'n/a')}."]
        for v in hb["verdicts"]:
            u = unc.get(v)
            if u:
                lines.append(f"Uncapped, `{v}` would have had coverage {fmt(u.get('gate_coverage'))} at precision "
                             f"{fmt(u.get('gate_precision'))} on the evaluation split (n {fmt(u.get('gate_n'))}).")
    if rep.get("fit_split") == "val":
        lines += ["", "The map is fitted on the validation split, which also selected the checkpoint, so it is mildly "
                      "optimistic."]
    return "\n".join(lines) + "\n"


def _failure_modes(overrides):
    """Bullets from ``card_overrides.json``'s ``failure_modes``, or one TODO bullet."""
    modes = overrides.get("failure_modes")
    if isinstance(modes, str):
        modes = [modes]
    if modes:
        return "".join(f"- {m}\n" for m in modes)
    return (f"- {TODO} per-label-type failure modes from a review of this model's errors (e.g. the hard-example list); "
            "add them as `failure_modes` in card_overrides.json.\n")


def _doi_sentence(doi, what):
    """``DOI: ...`` when one is known, otherwise a plain statement (not a TODO: minting waits on final repo names)."""
    if doi:
        return f"DOI: [{doi}](https://doi.org/{doi})."
    return f"No DOI has been minted for this {what} yet."


def model_card(ctx):
    """The model repo's ``README.md``.

    Args:
        ctx: dict with keys ``label_type``, ``model_repo``, ``dataset_repo``, ``metrics`` (``metrics.json``),
            ``api_map`` and ``calibration_report`` (or None), ``trainer`` (trimmed ``trainer_state.json``),
            ``training_run`` and ``generation_run`` (dicts, possibly empty), ``dataset`` (``dataset_summary``
            output), ``code_commit`` (str or None) and ``overrides`` (``card_overrides.json``, see
            ``OVERRIDE_KEYS``).

    Returns:
        The card as a string (front matter + Markdown).

    Raises:
        ValueError: if ``training_run`` or ``generation_run`` records ``fixed_crop_half_px`` (the card's crop
            description would be wrong; see :func:`require_depth_sized`).
    """
    t = ctx["label_type"]
    require_depth_sized(ctx.get("training_run"), ctx.get("generation_run"))
    overrides = ctx.get("overrides") or {}
    metrics = ctx.get("metrics") or {}
    trainer = ctx.get("trainer") or {}
    run = ctx.get("training_run") or {}
    gen = ctx.get("generation_run") or {}
    ds = ctx.get("dataset") or {}
    test = metrics.get("test") or {}
    name = ctx["model_repo"].split("/")[-1]

    results = []
    for metric_type, key, label in (("accuracy", "accuracy", "Accuracy"),
                                    ("balanced_accuracy", "balanced_accuracy", "Balanced accuracy"),
                                    ("roc_auc", "auc", "AUC (incorrect)")):
        if test.get(key) is not None:
            results.append({"type": metric_type, "value": round(float(test[key]), 4), "name": label})
    if macro_f1(test) is not None:
        results.append({"type": "f1", "value": round(macro_f1(test), 4), "name": "Macro F1"})
    meta = {"license": "mit", "library_name": "transformers", "pipeline_tag": "image-classification",
            "base_model": run.get("base_checkpoint") or "facebook/dinov2-large",
            "datasets": [ctx["dataset_repo"]],
            "tags": ["project-sidewalk", "accessibility", "sidewalks", "street-view", "dinov2", "label-validation",
                     t.lower()],
            "metrics": ["accuracy", "roc_auc", "f1", "precision", "recall"]}
    if results:
        meta["model-index"] = [{"name": name, "results": [{
            "task": {"type": "image-classification", "name": "Label validation"},
            "dataset": {"type": ctx["dataset_repo"], "name": f"{name} dataset", "split": "test"},
            "metrics": results}]}]

    tilt = gen.get("tilt_beta", run.get("tilt_beta"))
    encoder = gen.get("depth_encoder", run.get("depth_encoder"))
    best = trainer.get("best_model_checkpoint") or run.get("best_model_checkpoint")
    best_metric = run.get("best_metric") or "macro_f1"
    best_value = trainer.get("best_metric", run.get("best_metric_value"))
    split_rows = ds.get("split_counts") or {}
    heldout = ds.get("heldout_cities") or []

    parts = [yaml_front_matter(meta), f"# {name}\n",
             f"A binary image classifier that checks one crowdsourced [Project Sidewalk](https://projectsidewalk.org) "
             f"**{t}** label: given a crop of street-level imagery centred on the label, it predicts whether the "
             f"label is **correct** (the image shows {LABEL_TYPE_DESCRIPTIONS.get(t, t)}) or **incorrect**. It is "
             f"`facebook/dinov2-large` fine-tuned with a two-class head (`id2label = {{0: correct, 1: incorrect}}`) "
             f"by [sidewalk-validator-ai]({CODE_URL}).\n",
             "## Intended use\n",
             f"Automatic validation of new {t} labels in Project Sidewalk. [sidewalk-ai-api]({API_URL}) rebuilds the "
             f"crop from the panorama and label point, runs this model, and looks the verdict's confidence up in "
             f"`confidence_precision_map.json`. [SidewalkWebpage]({PLATFORM_URL}) records the verdict as an "
             f"Agree/Disagree vote by the `SidewalkAI` account only when that estimated accuracy is at least "
             f"**{GATE}** (`ai-validation-min-accuracy`); otherwise it is stored as Unsure. The model is one input "
             f"to label quality, not a replacement for human validation.\n",
             "Out of scope: finding features in whole panoramas (that is RampNet's job), other label types, and "
             "imagery that is not a label-centred crop made the same way as the training crops.\n",
             "## Training data\n",
             f"[`{ctx['dataset_repo']}`](https://huggingface.co/datasets/{ctx['dataset_repo']}) "
             f"(content hash `{ds.get('content_hash', TODO + ' dataset not staged')}`). Labels are taken as correct "
             f"or incorrect from **human** validation votes only (AI votes excluded); see the dataset card for the "
             f"exact thresholds. Split sizes: train {fmt(split_rows.get('train'))}, val {fmt(split_rows.get('val'))}, "
             f"test {fmt(split_rows.get('test'))}"
             + (f", test_heldout {fmt(split_rows.get('test_heldout'))} ({', '.join(heldout)})" if heldout else "")
             + ". The split is grouped by panorama (no panorama in two splits), stratified by class, seeded, with "
               "optional whole held-out cities (`split_data.py`).\n",
             "## How the input crop is made\n",
             f"The panorama is resized to 8192×4096. A {dg.VIEW_SIZE}² perspective view with a {dg.VIEW_FOV_DEG:g}° "
             f"field of view is rendered **centred on the label** (`sidewalk-panorama-tools`), with rig-tilt "
             f"correction from the label's camera pitch and roll (beta = {fmt(tilt, 2) if tilt is not None else TODO + ' tilt beta'}). "
             f"Depth Anything V2 (metric, {dg.DEPTH_DATASET.upper()}, "
             f"{ENCODER_NAMES.get(encoder, encoder) if encoder else TODO + ' encoder'}) estimates the depth at the view centre; the crop "
             f"half-width is `{dg.CROP_DEPTH_CONSTANT} / depth_m` pixels, clipped to the view, and the crop's long "
             f"side is resized to {dg.CROP_LONG_SIDE} px (training crops are stored as WebP quality "
             f"{dg.WEBP_QUALITY}).\n",
             "**Deployment order.** The model expects tilt-corrected crops, so the serving API must apply the same "
             "correction ([sidewalk-ai-api#17](https://github.com/ProjectSidewalk/sidewalk-ai-api/pull/17)).\n",
             "## Preprocessing and training\n",
             "Inference uses this repo's `preprocessor_config.json` (`BitImageProcessor`: resize shortest edge 256, "
             "centre-crop 224, ImageNet mean/std); load it with `use_fast=False` to match the API's PIL path. "
             "Training adds `RandomResizedCrop(224, scale=(0.7, 1.0))` and horizontal flips; validation uses the "
             f"processor itself. Hugging Face `Trainer`, learning rate 1e-5, batch {TRAIN_BATCH_SIZE} × {GRAD_ACCUM} "
             "gradient accumulation, "
             f"10% warm-up, {fmt(trainer.get('num_train_epochs') or run.get('epochs'))} epochs, seed "
             f"{fmt(run.get('seed')) if run.get('seed') is not None else TODO + ' seed (training_run.json missing)'}. "
             f"The published weights are the checkpoint with the best validation {best_metric} "
             f"({fmt(best_value)}; `{best or TODO + ' best checkpoint'}`); `trainer_state.json` keeps the full log "
             "history.\n",
             "```python\n"
             "from PIL import Image\n"
             "from transformers import AutoImageProcessor, AutoModelForImageClassification\n\n"
             f"processor = AutoImageProcessor.from_pretrained(\"{ctx['model_repo']}\", use_fast=False)\n"
             f"model = AutoModelForImageClassification.from_pretrained(\"{ctx['model_repo']}\").eval()\n"
             "probs = model(**processor(Image.open(\"crop.webp\").convert(\"RGB\"), return_tensors=\"pt\"))"
             ".logits.softmax(-1)[0]\n"
             "verdict = model.config.id2label[int(probs.argmax())]   # \"correct\" or \"incorrect\"\n"
             "```\n",
             "## Evaluation\n",
             "From `metrics.json` (`evaluate.py`), on the best checkpoint. The verdict is the argmax with ties going "
             "to `correct`, as in the API.\n",
             _metrics_section(metrics),
             "## Calibration\n",
             _calibration_section(ctx),
             "## Known failure modes and limitations\n",
             "- Trained and evaluated only on Google Street View panoramas from the cities listed in the dataset "
             "card; performance elsewhere, and on other imagery, is unmeasured beyond the held-out cities above.\n"
             "- Test metrics are on the training class mix (about 55:45 correct:incorrect). Production traffic is "
             "mostly correct labels, so precision on `incorrect` verdicts is lower in deployment than the test "
             "table suggests; the calibrated map accounts for this, the raw metrics do not.\n"
             "- The `hard` slice is small; read its numbers with wide error bars.\n"
             "- Correct/incorrect comes from crowd votes, which are themselves noisy; only decisively voted labels "
             "are used, so borderline cases are under-represented.\n"
             "- A crop made without the tilt correction, or with a different depth model, is out of distribution.\n"
             + _failure_modes(overrides),
             "## Files\n",
             "| file | contents |\n|---|---|\n"
             "| `model.safetensors`, `config.json` | weights and `Dinov2ForImageClassification` config |\n"
             "| `preprocessor_config.json` | the image processor used in validation, evaluation and serving |\n"
             "| `confidence_precision_map.json` | confidence → estimated accuracy, per verdict (the API's format) |\n"
             "| `calibration_report.json` | prior, ECE and gate coverage of that map (when present) |\n"
             "| `metrics.json` | the evaluation above |\n"
             "| `trainer_state.json` | trimmed: log history and best checkpoint only |\n"
             "| `training_run.json` | seed, epochs, package versions, generation settings (when present) |\n",
             "## Lineage\n",
             LINEAGE + "\n",
             f"Published from [sidewalk-validator-ai]({CODE_URL})"
             + (f" at commit `{ctx['code_commit']}`" if ctx.get("code_commit") else "")
             + " with `publish_to_hf.py`.\n",
             "## Citation\n",
             _doi_sentence(overrides.get("model_doi"), "model") + "\n",
             "There is no paper for this model yet. Please cite Project Sidewalk and link the training repository:\n",
             "```bibtex\n" + CITATION_BIBTEX + "\n```\n"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------------------------------------------------

def _counts_table(ds):
    counts = ds.get("counts") or {}
    lines = ["| split | correct | incorrect | total | share incorrect |", "|---|---|---|---|---|"]
    for split in ds.get("splits", []):
        c = counts.get(split, {})
        total = c.get("correct", 0) + c.get("incorrect", 0)
        share = c.get("incorrect", 0) / total if total else None
        lines.append(f"| {split} | {fmt(c.get('correct', 0))} | {fmt(c.get('incorrect', 0))} | {fmt(total)} | "
                     f"{fmt(share)} |")
    tc = ds.get("class_totals", {})
    lines.append(f"| **all** | {fmt(tc.get('correct', 0))} | {fmt(tc.get('incorrect', 0))} | {fmt(ds.get('n'))} | "
                 f"{fmt(tc.get('incorrect', 0) / ds['n'] if ds.get('n') else None)} |")
    return "\n".join(lines) + "\n"


def _cities_table(ds):
    rows = ds.get("cities") or {}
    heldout = set(ds.get("heldout_cities") or [])
    lines = ["| city | crops | in test_heldout |", "|---|---|---|"]
    for city, n in rows.items():
        lines.append(f"| {city} | {fmt(n)} | {'yes' if city in heldout else ''} |")
    return "\n".join(lines) + "\n"


def dataset_card(ctx, license_id=None):
    """The dataset repo's ``README.md``.

    Args:
        ctx: dict with keys ``label_type``, ``dataset_repo``, ``model_repo``, ``dataset`` (``dataset_summary``
            output, including ``content_hash``), ``prior`` (this label type's ``deployment_priors.json`` entry or
            None), ``generation_run`` (dict), ``code_commit`` and ``overrides`` (see ``OVERRIDE_KEYS``).
        license_id: Hub license identifier for the dataset (``--dataset-license``, or ``dataset_license`` in the
            overrides). None writes a ``TODO:`` (the dataset license has not been decided; issue #2) and leaves
            ``license`` out of the metadata rather than guessing.

    Raises:
        ValueError: if ``generation_run`` records ``fixed_crop_half_px`` (see :func:`require_depth_sized`).
    """
    t = ctx["label_type"]
    require_depth_sized(ctx.get("generation_run"))
    overrides = ctx.get("overrides") or {}
    license_id = license_id or overrides.get("dataset_license")
    split_seed = overrides.get("split_seed")
    ds = ctx["dataset"]
    gen = ctx.get("generation_run") or {}
    settings = gen.get("settings") or {}
    prior = ctx.get("prior")
    name = ctx["dataset_repo"].split("/")[-1]
    split_dirs = {"train": "train", "val": "validation", "test": "test", "test_heldout": "test_heldout"}

    meta = {}
    comments = []
    if license_id:
        meta["license"] = license_id
    else:
        comments.append(f"{TODO} dataset license (issue #2); pass --dataset-license once decided")
    meta.update({"task_categories": ["image-classification"], "size_categories": [size_category(ds["n"])],
                 "pretty_name": f"Project Sidewalk {t} label validation crops",
                 "tags": ["project-sidewalk", "accessibility", "sidewalks", "street-view", "crowdsourcing",
                          "label-validation", t.lower()],
                 "configs": [{"config_name": "default",
                              "data_files": [{"split": split_dirs[s], "path": f"{s}/*/*.webp"}
                                             for s in ds.get("splits", [])]}]})

    years = ds.get("capture_years") or {}
    year_text = (f"{years['min']}–{years['max']}" if years.get("min") is not None
                 else f"{TODO} no image_capture_date values in the manifest")
    include_ai = settings.get("include_ai_labels", gen.get("include_ai_labels"))
    n_experts = gen.get("n_experts") or 0
    expert_row = ("| either | otherwise, the class an expert account voted for, if every expert vote on the label agrees "
                  "and no other human voted the other way |\n")
    source_counts = gen.get("class_source_counts") or {}
    expert_text = (f"\nExpert votes came from {n_experts} project-team account{'s' if n_experts != 1 else ''}; the "
                   "manifest's `class_source` column says which rule decided each label (`margin` or `expert`)"
                   + (f": {fmt(source_counts.get('margin', 0))} margin and {fmt(source_counts.get('expert', 0))} "
                      "expert in the selection." if source_counts else ".") + "\n")
    rule = " margin" if n_experts else ""  # the prior counts margin-rule labels only
    prior_text = (f"Before downsampling, {fmt(prior.get('correct'))} labels met the `correct`{rule} rule and "
                  f"{fmt(prior.get('incorrect'))} the `incorrect`{rule} rule "
                  f"({fmt(prior['correct'] / (prior['correct'] + prior['incorrect']))} correct; "
                  f"{prior.get('source', '')}, measured {prior.get('measured_on', 'n/a')}). That is the deployment "
                  f"prior the served calibration map is reweighted to."
                  if prior else f"{TODO} deployment_priors.json has no entry for {t}.")
    heldout = ds.get("heldout_cities") or []
    selection = gen.get("selection_sha256")

    parts = [yaml_front_matter(meta, comments), f"# {name}\n",
             f"Street-level image crops of crowdsourced [Project Sidewalk](https://projectsidewalk.org) **{t}** "
             f"labels, each marked **correct** or **incorrect** by the crowd's validation votes. It is the training "
             f"and evaluation data of [`{ctx['model_repo']}`](https://huggingface.co/{ctx['model_repo']}), built by "
             f"[sidewalk-validator-ai]({CODE_URL}) (`data_generation.py`, `split_data.py`).\n",
             f"- **Crops:** {fmt(ds['n'])} WebP images, one per label, laid out as "
             "`<split>/<correct|incorrect>/<city>_<label_id>.webp`.\n"
             f"- **Cities:** {len(ds.get('cities') or {})}; **image capture years:** {year_text}.\n"
             f"- **Panoramas:** {fmt(ds.get('n_panoramas'))}.\n"
             f"- **Content hash:** `{ds['content_hash']}` (sha256 of `SHA256SUMS`, which lists the sha256 of every "
             "crop by path; recompute it to check an upload).\n"
             f"- **License:** {license_id or TODO + ' to be decided (issue #2)'}.\n",
             "## How labels become correct or incorrect\n",
             "Every label of this type and its validations are pulled from each deployment's public "
             "`/v3/api/rawLabels` export. Only **human** votes are counted: each vote carries a `validator_type`, and "
             "votes cast by the `SidewalkAI` account are excluded, so the model is never trained on its own output.\n",
             "| class | rule (human votes) |\n|---|---|\n"
             "| `incorrect` | disagree − agree > 1 |\n"
             "| `correct` | agree − disagree > 2, then randomly downsampled |\n"
             + (expert_row if n_experts else "")
             + "| (dropped) | unvalidated, unsure, or contested |\n"
             + (expert_text if n_experts else ""),
             f"Labels placed by the `SidewalkAI` account are "
             f"{'included (tagged `author = ai` in the manifest)' if include_ai else 'excluded'}; labels from "
             "non-GSV imagery, with unusable panorama geometry, or with legacy unconverted coordinates are excluded. "
             f"The `correct` class is then downsampled to {dg.CORRECT_PROPORTION:.0%} of the dataset by a seeded hash "
             f"of `label_uid` (seed {gen.get('seed', dg.SEED)}), a training-balance choice, not the real class mix. "
             + prior_text + "\n",
             "## How a crop is made\n",
             f"The panorama is read from the lab's panorama store and resized to 8192×4096. A {dg.VIEW_SIZE}² "
             f"perspective view with a {dg.VIEW_FOV_DEG:g}° field of view is rendered centred on the label with "
             f"rig-tilt correction (beta = {settings.get('tilt_beta', gen.get('tilt_beta', TODO + ' tilt beta'))}). "
             f"Depth Anything V2 (metric, "
             f"{str(settings.get('depth_dataset', dg.DEPTH_DATASET)).upper()}, "
             f"{ENCODER_NAMES.get(settings.get('depth_encoder', gen.get('depth_encoder')), TODO + ' encoder')}) estimates "
             f"the depth at the view centre, the crop half-width is `{dg.CROP_DEPTH_CONSTANT} / depth_m` pixels "
             f"(clipped to the view; `clipped = 1` in the manifest), and the crop's long side is resized to "
             f"{dg.CROP_LONG_SIDE} px and saved as WebP quality {dg.WEBP_QUALITY}.\n",
             "## Splits\n",
             _counts_table(ds),
             "`split_data.py` groups labels by panorama so that no panorama appears in two splits (overlapping crops "
             "of one corner would otherwise leak), bins the groups by class ratio and fills train/val/test 70/15/15 "
             "of rows per bin with a seeded shuffle"
             + (f"; every label from {', '.join(heldout)} is in `test_heldout`" if heldout else
                "; no cities were held out")
             + (f" (split seed {split_seed})" if split_seed is not None else
                f". {TODO} split seed (split_data.py does not record it; its default is 42; add `split_seed` to "
                "card_overrides.json)")
             + ". The Hub config maps "
               "`val` to the `validation` split.\n",
             "## Cities\n",
             _cities_table(ds),
             "## Files\n",
             "| file | contents |\n|---|---|\n"
             "| `<split>/<class>/<city>_<label_id>.webp` | the crops |\n"
             "| `split.csv` | one row per crop: `label_uid` (`<city>:<label_id>`, the unique key; `label_id` alone "
             "repeats across cities), votes, pose, depth, crop size, `hard`, `split`, and `crop_path` within this "
             "repo |\n"
             "| `manifest.csv` | every label the generator attempted, including failures (`status`) |\n"
             "| `deployment_priors.json` | pre-downsampling class counts for this label type |\n"
             "| `generation_run.json` | generator settings and the selection hash |\n"
             "| `dataset_info.json` | the counts on this card, machine-readable |\n"
             "| `SHA256SUMS` | sha256 of every crop |\n"
             "| `croissant.json` | Croissant 1.1 metadata |\n",
             "The absolute panorama-store paths recorded at generation time (`pano_path`, `pano_store`, `pano_store_marker`) are "
             "removed from the published copies.\n",
             "## Intended use\n",
             f"Training and evaluating validators for Project Sidewalk {t} labels, and research on crowd label "
             "quality. Not suitable for estimating how common a feature is in a city: labels exist where volunteers "
             "chose to label, and the classes are rebalanced.\n",
             "## Versioning\n",
             "Each upload is a commit of this repo; the content hash above identifies the crops exactly, and "
             f"`generation_run.json` records the selection hash (`{selection or TODO + ' selection hash'}`), which "
             "changes when any label enters, leaves or changes class. A regenerated dataset is uploaded as a new "
             "commit with a new card. " + _doi_sentence(overrides.get("doi"), "dataset") + "\n",
             "## Citation\n",
             "There is no paper for this dataset yet. Please cite Project Sidewalk and link the repository:\n",
             "```bibtex\n" + CITATION_BIBTEX + "\n```\n",
             f"Published from [sidewalk-validator-ai]({CODE_URL})"
             + (f" at commit `{ctx['code_commit']}`" if ctx.get("code_commit") else "")
             + " with `publish_to_hf.py`.\n"]
    return "\n".join(parts)
