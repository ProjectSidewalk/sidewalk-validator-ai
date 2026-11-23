"""Croissant RAI metadata for the published dataset, and a lint for it (issue #2).

``publish_to_hf.py`` builds ``croissant.json`` (Croissant 1.1) at staging; this module adds the Responsible AI
fields of the Croissant RAI 1.0 extension (https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html) and
checks them. Pure functions: nothing here reads the pipeline root or imports ``publish_to_hf``.

Where the text comes from: the redacted dataset summary, the **staged** ``split.csv`` and ``manifest.csv`` (after the
private deployments are dropped), the prior and generation run exactly as the dataset card receives them, and the
card's own sentences (``hf_cards``). Nothing is guessed and no ``TODO:`` is written: a sentence whose input is missing
is left out, and a property with nothing left is omitted. Operators can replace any value through the ``rai`` key of
``card_overrides.json`` (keys in the prefixed form, ``"rai:dataBiases"``); a restage applies it.

Why :func:`lint` exists: ``mlcroissant`` 1.1.1 accepts an unknown ``rai:`` key without a word, only warns on a list
given to a single-valued property, and for two properties spells the name differently from the RAI vocabulary
(``dataCollectionTimeFrame``, ``dataDataManipulationProtocol``). Those two properties are not written at all.

Example::

    import hf_croissant
    rai = hf_croissant.rai_metadata(ctx, split_df, manifest_df)   # ctx: the dataset card's context dict
    doc.update(rai)
    problems = hf_croissant.lint(doc)                             # [] when the document is fine
"""
import re

import hf_cards

CROISSANT_URI = "http://mlcommons.org/croissant/1.1"
RAI_URI = "http://mlcommons.org/croissant/RAI/1.0"
CONFORMS_TO = [CROISSANT_URI, RAI_URI]
"""``conformsTo`` of the dataset's ``croissant.json`` (Croissant 1.1 allows a list; the core URI must stay)."""

RAI_PROPERTIES = {  # name: (cardinality in the RAI 1.0 spec table, cardinality in mlcroissant 1.1.1)
    "dataCollection": ("ONE", "ONE"),
    "dataCollectionType": ("MANY", "MANY"),
    "dataCollectionMissingData": ("ONE", "ONE"),
    "dataCollectionRawData": ("ONE", "ONE"),
    "dataImputationProtocol": ("ONE", "ONE"),
    "dataPreprocessingProtocol": ("MANY", "MANY"),
    "dataAnnotationProtocol": ("ONE", "MANY"),
    "dataAnnotationPlatform": ("MANY", "MANY"),
    "dataAnnotationAnalysis": ("MANY", "MANY"),
    "annotationsPerItem": ("ONE", "ONE"),
    "annotatorDemographics": ("MANY", "MANY"),
    "machineAnnotationTools": ("MANY", "MANY"),
    "dataBiases": ("MANY", "MANY"),
    "dataUseCases": ("MANY", "MANY"),
    "dataLimitations": ("MANY", "MANY"),
    "dataSocialImpact": ("ONE", "ONE"),
    "personalSensitiveInformation": ("MANY", "MANY"),
    "dataReleaseMaintenancePlan": ("MANY", "ONE"),
}
"""The 18 RAI properties written. The spec's other two (``dataCollectionTimeframe`` and ``dataManipulationProtocol``)
are left out: ``mlcroissant`` 1.1.1 parses them only under other spellings and ignores the spec's silently."""

SPELLING_CONFLICTS = ("dataCollectionTimeframe", "dataCollectionTimeFrame", "dataManipulationProtocol",
                      "dataDataManipulationProtocol")

LICENCE_STEMS = (r"licen[cs]", r"copyright", r"terms of (use|service)", r"redistribut", r"permission", r"proprietary",
                 r"scrap")
"""Case-insensitive patterns :func:`lint` refuses in RAI text: licence statements belong in the ``license`` field."""

CARD_TO_RAI_REWRITES = {"the content hash above": "the content hash in this file's description"}
"""Card phrases that point at the card's own layout, and what the RAI text says instead."""

COLLECTION_TYPES = ["User-generated content data", "Web API"]

IMPUTATION = "No values are imputed; blank cells are missing values."
ANNOTATION_PLATFORM = ("Project Sidewalk's validation interface (https://projectsidewalk.org), in which users vote "
                       "Agree, Disagree or Unsure on labels.")
ANNOTATOR_DEMOGRAPHICS = "No annotator demographics or identifiers are included in this dataset."
PERSONAL_SENSITIVE = ("Crops are street-level imagery that can show pedestrians, vehicles (number plates included) "
                      "and building fronts.")
SOCIAL_IMPACT = ("The data is for training and evaluating validators of crowdsourced sidewalk accessibility labels. "
                 "A validator's errors can hide correct labels or keep wrong ones in the accessibility data built "
                 "from them, so it is one input to label quality, not a replacement for human validation.")
AI_VOTES_BIAS = "Votes by the AI validator account are not counted, but they can affect which labels people are later shown."


# ---------------------------------------------------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------------------------------------------------

def plain_text(text):
    """Markdown to the plain text RAI values hold: links become their text (an issue link its URL), bare issue
    references become full URLs, backticks and bold marks go, whitespace collapses.

    >>> plain_text("**human** votes; see `split.csv` and [#42](https://x/issues/42), or issue #56")
    'human votes; see split.csv and https://x/issues/42, or issue https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56'
    """
    text = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", lambda m: m.group(2) if m.group(1).startswith("#") else m.group(1),
                  str(text))
    text = re.sub(r"(?<![\w/\[])#(\d+)\b", lambda m: f"{hf_cards.CODE_URL}/issues/{m.group(1)}", text)
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def sentences(text):
    """Split plain text into sentences (at ``.``, ``!`` or ``?`` followed by a space and a capital or bracket).

    >>> sentences("One (a; b). Two is here. TODO: three.")
    ['One (a; b).', 'Two is here.', 'TODO: three.']
    """
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text) if s]


def _capitalise(text):
    return text[:1].upper() + text[1:]


def known(text):
    """``text`` as plain text without the sentences that hold a ``TODO:`` marker (an input the card says is
    missing), or "".

    >>> known("Seed 42. Beta = TODO: tilt beta. Done.")
    'Seed 42. Done.'
    """
    return " ".join(s for s in sentences(plain_text(text)) if hf_cards.TODO not in s)


def _join(*parts):
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------------------------------

def vote_stats(split_df):
    """Min, median and max of Agree plus Disagree human votes per label in ``split_df``, or None when it has no vote
    columns or no rows. Blank vote cells count as 0.

    >>> import pandas as pd
    >>> vote_stats(pd.DataFrame({"human_agree": [3, 4, 6], "human_disagree": [0, 1, None]}))
    {'min': 3, 'median': 5, 'max': 6}
    """
    import pandas as pd
    if split_df is None or not {"human_agree", "human_disagree"} <= set(split_df.columns) or not len(split_df):
        return None
    total = (pd.to_numeric(split_df["human_agree"], errors="coerce").fillna(0)
             + pd.to_numeric(split_df["human_disagree"], errors="coerce").fillna(0))
    median = float(total.median())
    return {"min": int(total.min()), "median": int(median) if median.is_integer() else median,
            "max": int(total.max())}


def _status_counts(manifest_df):
    """``"ok 12, no_pano 1"`` from the staged manifest's ``status`` column (most common first), or ""."""
    if manifest_df is None or "status" not in manifest_df.columns:
        return ""
    counts = manifest_df["status"].fillna("").astype(str).value_counts()
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k or 'blank'} {int(v)}" for k, v in rows)


def _collection(ds):
    cities = ds.get("cities") or {}
    years = ds.get("capture_years") or {}
    sources = ds.get("pano_sources") or {}
    city_text = (f"Labels and validation votes from {len(cities)} Project Sidewalk "
                 f"{'city' if len(cities) == 1 else 'cities'} ({', '.join(cities)}), read from each deployment's "
                 "public /v3/api/rawLabels export." if cities else "")
    year_text = (f"Image capture years of the panoramas: {years['min']}–{years['max']}."
                 if years.get("min") is not None else "")
    source_text = ("Imagery provider per label: "
                   + ", ".join(f"{k or 'not recorded (GSV, older manifests)'} {v}" for k, v in sources.items()) + "."
                   if sources else "")
    return _join(city_text, year_text, source_text)


def _missing_data(manifest_df):
    counts = _status_counts(manifest_df)
    return _join("Labels the class rules leave out (unvalidated, unsure or contested) are not in this dataset.",
                 f"manifest.csv lists every label the generator attempted, with its status: {counts}; only ok rows "
                 "have a crop." if counts else "")


def _annotation_protocol(n_experts):
    rules = "; ".join(f"{plain_text(cls).strip('()')}{' class' if cls == 'either' else ''}: {plain_text(rule)}"
                      for cls, rule in hf_cards.vote_rules(n_experts))
    return _join(plain_text(hf_cards.VOTE_SOURCE_TEXT), f"Class rules over human votes: {rules}.")


def _annotation_analysis(prior, n_experts, source_counts):
    return _join(known(hf_cards.expert_text(n_experts, source_counts)),
                 known(hf_cards.prior_sentence(prior, n_experts)) if prior else "")


def _annotations_per_item(stats):
    if not stats:
        return ""
    count = (f"{stats['min']} Agree plus Disagree votes on every label" if stats["min"] == stats["max"] else
             f"{stats['min']}–{stats['max']} Agree plus Disagree votes per label (median {stats['median']})")
    return f"{count}, counting human votes on labels that passed a class rule; Unsure votes are not in the file."


def _machine_tools(gen, include_ai):
    settings = gen.get("settings") or {}
    encoder = settings.get("depth_encoder", gen.get("depth_encoder"))
    depth = (f"Depth Anything V2 (metric, {str(settings.get('depth_dataset', hf_cards.dg.DEPTH_DATASET)).upper()}, "
             f"{hf_cards.ENCODER_NAMES[encoder]}) estimates the depth that sizes each crop; no model assigns the "
             "classes, which come from human votes." if encoder in hf_cards.ENCODER_NAMES else "")
    return _join(depth, plain_text(hf_cards.ai_labels_text(include_ai)) + ".")


def _biases(ds):
    cities = ds.get("cities") or {}
    mix = (f"The cities contribute unequal numbers of crops ({', '.join(f'{c} {n}' for c, n in cities.items())})."
           if len(cities) > 1 else "")
    return _join("Labels exist only where Project Sidewalk users chose to label, so coverage follows where users "
                 "worked, not where features are.", mix,
                 "Only labels with a decisive human vote are included, so borderline cases are under-represented.",
                 AI_VOTES_BIAS)


def _maintenance(selection, doi):
    text = known(hf_cards.versioning_text(selection, doi))
    for old, new in CARD_TO_RAI_REWRITES.items():
        text = text.replace(old, new)
    return text


def rai_metadata(ctx, split_df, manifest_df):
    """The ``rai:*`` properties for a staged dataset, in :data:`RAI_PROPERTIES` order.

    Args:
        ctx: the dataset card's context (``label_type``, ``dataset`` = the redacted ``dataset_summary``, ``prior``,
            ``generation_run``, ``overrides``), as ``publish_to_hf.stage`` passes it to ``hf_cards.dataset_card``.
        split_df: the staged ``split.csv``.
        manifest_df: the staged ``manifest.csv`` (private deployments already dropped).

    Returns:
        ``{"rai:<name>": text}``; ``rai:dataCollectionType`` is the one list. A property with no known text is
        omitted. Overrides are applied separately (:func:`apply_overrides`).
    """
    t = ctx["label_type"]
    ds = ctx.get("dataset") or {}
    gen = ctx.get("generation_run") or {}
    settings = gen.get("settings") or {}
    overrides = ctx.get("overrides") or {}
    n_experts = gen.get("n_experts") or 0
    include_ai = settings.get("include_ai_labels", gen.get("include_ai_labels"))
    use = sentences(known(hf_cards.intended_use_text(t)))
    values = {
        "dataCollection": _collection(ds),
        "dataCollectionType": list(COLLECTION_TYPES),
        "dataCollectionMissingData": _missing_data(manifest_df),
        "dataCollectionRawData": ("Labels and their validation votes come from each deployment's public "
                                  "/v3/api/rawLabels export; the panoramas are read from the lab's panorama store."),
        "dataImputationProtocol": IMPUTATION,
        "dataPreprocessingProtocol": _join(
            _capitalise(known(hf_cards.imagery_sentence(ds.get("pano_sources") or {}, settings))),
            known(hf_cards.downsample_text(gen)), known(hf_cards.crop_text(ds, gen)),
            known(hf_cards.split_text(ds.get("heldout_cities") or [], overrides.get("split_seed")))),
        "dataAnnotationProtocol": _annotation_protocol(n_experts),
        "dataAnnotationPlatform": ANNOTATION_PLATFORM,
        "dataAnnotationAnalysis": _annotation_analysis(ctx.get("prior"), n_experts,
                                                       gen.get("class_source_counts") or {}),
        "annotationsPerItem": _annotations_per_item(vote_stats(split_df)),
        "annotatorDemographics": ANNOTATOR_DEMOGRAPHICS,
        "machineAnnotationTools": _machine_tools(gen, include_ai),
        "dataBiases": _biases(ds),
        "dataUseCases": _join(*use[:1]),
        "dataLimitations": _join(*use[1:], "label_id is unique only within a city; key rows on label_uid "
                                           "(<city>:<label_id>)."),
        "dataSocialImpact": SOCIAL_IMPACT,
        "personalSensitiveInformation": PERSONAL_SENSITIVE,
        "dataReleaseMaintenancePlan": _maintenance(gen.get("selection_sha256"), overrides.get("doi")),
    }
    out = {}
    for name in RAI_PROPERTIES:
        v = values.get(name)
        if isinstance(v, str):
            v = _capitalise(v.strip())
        if v:
            out[f"rai:{name}"] = v
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Overrides and lint
# ---------------------------------------------------------------------------------------------------------------------

def apply_overrides(rai, override):
    """``(merged, problems)``: ``rai`` with the ``rai`` entry of ``card_overrides.json`` laid over it. Keys must be the
    prefixed form (``"rai:dataBiases"``); a bare key is a problem that names the prefixed one. Unknown prefixed keys
    and bad values are left to :func:`lint`, which runs on the merged document.

    >>> apply_overrides({"rai:dataBiases": "a"}, {"dataBiases": "b"})[1]
    ["card_overrides.json 'rai': key 'dataBiases' must be written 'rai:dataBiases'"]
    """
    merged = dict(rai)
    if override is None:
        return merged, []
    if not isinstance(override, dict):
        return merged, ["card_overrides.json 'rai' must be an object of 'rai:<property>': text"]
    problems = []
    for key, value in override.items():
        if not str(key).startswith("rai:"):
            problems.append(f"card_overrides.json 'rai': key {key!r} must be written 'rai:{key}'")
            continue
        merged[key] = value
    return merged, problems


def list_allowed(name):
    """A list is allowed only where the RAI spec and mlcroissant 1.1.1 both say MANY."""
    return RAI_PROPERTIES.get(name) == ("MANY", "MANY")


def lint(doc):
    """Problems with the RAI part of a Croissant document, as a list of strings ([] when there are none).

    Checks: ``conformsTo`` lists both :data:`CONFORMS_TO` URIs; every ``rai:`` key is in :data:`RAI_PROPERTIES`
    (so both spellings of the two conflicting properties are refused); a list only where :func:`list_allowed`; every
    value a non-empty string with no ``TODO:``; no licence statement (:data:`LICENCE_STEMS`).

    >>> lint({"conformsTo": CONFORMS_TO, "rai:dataBiases": "Fine."})
    []
    """
    problems = []
    conforms = doc.get("conformsTo")
    if not isinstance(conforms, list) or not set(CONFORMS_TO) <= set(conforms):
        problems.append(f"conformsTo must list both {CROISSANT_URI} and {RAI_URI}, not {conforms!r}")
    for key, value in doc.items():
        if not str(key).startswith("rai:"):
            continue
        name = key[len("rai:"):]
        if name not in RAI_PROPERTIES:
            why = (" (left out on purpose: the RAI spec and mlcroissant 1.1.1 spell it differently)"
                   if name in SPELLING_CONFLICTS else "")
            problems.append(f"{key}: unknown RAI property{why}; allowed: {', '.join(RAI_PROPERTIES)}")
            continue
        if isinstance(value, list):
            if not list_allowed(name):
                problems.append(f"{key}: a list, but the RAI spec and mlcroissant 1.1.1 do not both allow several "
                                f"values here; write one string")
            if not value:
                problems.append(f"{key}: empty value")
            items = value
        else:
            items = [value]
        for v in items:
            if not isinstance(v, str):
                problems.append(f"{key}: values must be strings, not {type(v).__name__}")
                continue
            if not v.strip():
                problems.append(f"{key}: empty value")
            if hf_cards.TODO in v:
                problems.append(f"{key}: holds a {hf_cards.TODO} marker")
            for stem in LICENCE_STEMS:
                if re.search(stem, v, re.I):
                    problems.append(f"{key}: reads like a licence statement (matches {stem!r}); licence terms belong "
                                    f"in the license field, not in RAI text")
    return problems
