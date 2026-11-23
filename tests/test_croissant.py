"""Croissant metadata of the staged dataset (issue #2): the RAI block (``hf_croissant``), its lint, the column
descriptions, and what ``stage``/``verify_staged`` record and print about them.

Roots are built only through ``test_publish_to_hf.make_root`` (no network: its ``no_network`` fixture is imported).
Tests that need ``mlcroissant`` call ``conftest.require_mlcroissant()``: they skip without it, and fail without it
when ``REQUIRE_MLCROISSANT=1`` (as in CI).
"""
import json
import os
import re

import pandas as pd
import pytest

import data_generation as dg
import hf_cards
import publish_to_hf as pub
from test_publish_to_hf import LISTS, T, TEST_LISTS, make_root, no_network  # noqa: F401  (no_network is a fixture)

CROISSANT_URI = "http://mlcommons.org/croissant/1.1"
RAI_URI = "http://mlcommons.org/croissant/RAI/1.0"

# Explicit map from each emitted key to the attribute mlcroissant 1.1.1 parses it into. Comparing attributes (never
# to_json(), which echoes unknown keys verbatim) is what shows the library recognised a name.
MLCROISSANT_ATTRS = {
    "rai:dataCollection": "data_collection",
    "rai:dataCollectionType": "data_collection_type",
    "rai:dataCollectionMissingData": "data_collection_missing_data",
    "rai:dataCollectionRawData": "data_collection_raw_data",
    "rai:dataImputationProtocol": "data_imputation_protocol",
    "rai:dataPreprocessingProtocol": "data_preprocessing_protocol",
    "rai:dataAnnotationProtocol": "data_annotation_protocol",
    "rai:dataAnnotationPlatform": "data_annotation_platform",
    "rai:dataAnnotationAnalysis": "data_annotation_analysis",
    "rai:annotationsPerItem": "annotations_per_item",
    "rai:annotatorDemographics": "annotator_demographics",
    "rai:machineAnnotationTools": "machine_annotation_tools",
    "rai:dataBiases": "data_biases",
    "rai:dataUseCases": "data_use_cases",
    "rai:dataLimitations": "data_limitations",
    "rai:dataSocialImpact": "data_social_impact",
    "rai:personalSensitiveInformation": "personal_sensitive_information",
    "rai:dataReleaseMaintenancePlan": "data_release_maintenance_plan",
}


def _hc():
    import hf_croissant
    return hf_croissant


def _require_mlcroissant():
    from conftest import require_mlcroissant
    return require_mlcroissant()


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _rai(doc):
    return {k: v for k, v in doc.items() if k.startswith("rai:")}


def _values(doc):
    """Every RAI value as one list of strings."""
    out = []
    for v in _rai(doc).values():
        out.extend(v if isinstance(v, list) else [v])
    return out


def _rewrite_split(root, df, label_type=T):
    """Write ``split.csv`` and keep ``run_key.json``'s sha256 in step, so the binding check still passes."""
    base = os.path.join(root, label_type)
    df.to_csv(os.path.join(base, "split.csv"), index=False)
    key_path = os.path.join(base, "dinov2", "run_key.json")
    key = _read_json(key_path)
    key["split_sha256"] = pub.sha256_file(os.path.join(base, "split.csv"))
    with open(key_path, "w") as f:
        json.dump(key, f)


def _stage(tmp_path, **kw):
    root, *_ = make_root(tmp_path)
    model_dir, dataset_dir = pub.stage(root, T, lists=TEST_LISTS, **kw)
    return root, dataset_dir


def _croissant(dataset_dir):
    return _read_json(os.path.join(dataset_dir, "croissant.json"))


# ---------------------------------------------------------------------------------------------------------------------
# The RAI block as staged
# ---------------------------------------------------------------------------------------------------------------------

def test_full_fixture_writes_all_18_properties(tmp_path):
    _, d = _stage(tmp_path)
    rai = _rai(_croissant(d))
    assert set(rai) == {f"rai:{k}" for k in _hc().RAI_PROPERTIES}
    assert len(rai) == 18
    for k, v in rai.items():
        assert v, k
        for s in (v if isinstance(v, list) else [v]):
            assert isinstance(s, str) and s.strip(), k
            assert "TODO" not in s, k
            assert "`" not in s and "**" not in s, k      # plain text, not Markdown
    assert rai["rai:dataCollectionType"] == ["User-generated content data", "Web API"]
    assert [k for k, v in rai.items() if isinstance(v, list)] == ["rai:dataCollectionType"]


def test_a_root_with_no_prior_and_no_capture_dates_stages_without_todo(tmp_path):
    root, *_ = make_root(tmp_path)
    os.remove(os.path.join(root, "deployment_priors.json"))
    base = os.path.join(root, T)
    split = pd.read_csv(os.path.join(base, "split.csv"), dtype=str)
    split["image_capture_date"] = ""
    _rewrite_split(root, split)
    gen_path = os.path.join(base, "generation_run.json")
    gen = _read_json(gen_path)
    for k in ("tilt_beta", "depth_encoder", "selection_sha256"):
        gen.pop(k, None)
        gen["settings"].pop(k, None)
    with open(gen_path, "w") as f:
        json.dump(gen, f)
    _, d = pub.stage(root, T, lists=TEST_LISTS)
    card = open(os.path.join(d, "README.md")).read()
    assert "TODO:" in card                                  # the card still flags what is missing
    for v in _values(_croissant(d)):
        assert "TODO" not in v
    rai = _rai(_croissant(d))
    assert "capture years" not in rai["rai:dataCollection"]
    assert "deployment prior" not in rai.get("rai:dataAnnotationAnalysis", "")
    assert "beta" not in rai["rai:dataPreprocessingProtocol"]


def test_conforms_to_names_croissant_and_rai(tmp_path):
    _, d = _stage(tmp_path)
    assert _croissant(d)["conformsTo"] == [CROISSANT_URI, RAI_URI]


def test_licence_field_only_when_passed(tmp_path):
    _, d = _stage(tmp_path / "a")
    doc = _croissant(d)
    assert "license" not in doc and len(_rai(doc)) == 18
    _, d = _stage(tmp_path / "b", dataset_license="cc-by-4.0")
    doc = _croissant(d)
    assert doc["license"] == "cc-by-4.0" and len(_rai(doc)) == 18


def test_annotations_per_item_is_the_staged_vote_statistics(tmp_path):
    root, *_ = make_root(tmp_path)
    split = pd.read_csv(os.path.join(root, T, "split.csv"), dtype=str)
    split["human_agree"] = [str(3 + i) for i in range(len(split))]        # 3..14 Agree, 0 Disagree
    split.loc[0, "human_disagree"] = "1"
    _rewrite_split(root, split)
    _, d = pub.stage(root, T, lists=TEST_LISTS)
    stats = _hc().vote_stats(pd.read_csv(os.path.join(d, "split.csv")))
    assert stats == {"min": 4, "median": 8.5, "max": 14}
    text = _rai(_croissant(d))["rai:annotationsPerItem"]
    assert "4" in text and "8.5" in text and "14" in text
    assert "Agree plus Disagree" in text and "Unsure votes are not in the file" in text


# ---------------------------------------------------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------------------------------------------------

def _good_doc():
    return {"conformsTo": [CROISSANT_URI, RAI_URI], "rai:dataBiases": "Some bias.",
            "rai:dataCollectionType": ["User-generated content data", "Web API"]}


def test_lint_accepts_a_good_document():
    assert _hc().lint(_good_doc()) == []


@pytest.mark.parametrize("key, value, match", [
    ("rai:notAThing", "x", "unknown RAI property"),
    ("rai:dataCollectionTimeframe", "2020/2022", "unknown RAI property"),
    ("rai:dataCollectionTimeFrame", "2020/2022", "unknown RAI property"),
    ("rai:dataManipulationProtocol", "x", "unknown RAI property"),
    ("rai:dataDataManipulationProtocol", "x", "unknown RAI property"),
    ("rai:dataCollection", ["a", "b"], "list"),
    ("rai:dataAnnotationProtocol", ["a", "b"], "list"),          # spec ONE, mlcroissant MANY
    ("rai:dataReleaseMaintenancePlan", ["a", "b"], "list"),      # spec MANY, mlcroissant ONE
    ("rai:dataBiases", "", "empty"),
    ("rai:dataBiases", "   ", "empty"),
    ("rai:dataBiases", [], "empty"),
    ("rai:dataBiases", ["fine", ""], "empty"),
    ("rai:dataBiases", 3, "string"),
    ("rai:dataBiases", "TODO: write this", "TODO"),
])
def test_lint_rejects(key, value, match):
    doc = _good_doc()
    doc[key] = value
    problems = _hc().lint(doc)
    assert problems and any(re.search(match, p, re.I) for p in problems), problems


@pytest.mark.parametrize("text", [
    "Released under a licence.", "Licensed for research.", "Copyright 2026.",
    "See the terms of use.", "Subject to the Terms of Service.", "May not be redistributed.",
    "Used with permission.", "Proprietary data.", "Scraped from a website.",
])
def test_lint_rejects_licence_statements(text):
    doc = _good_doc()
    doc["rai:dataLimitations"] = text
    problems = _hc().lint(doc)
    assert problems and any("license field" in p for p in problems), problems


def test_lint_accepts_in_terms_of():
    doc = _good_doc()
    doc["rai:dataLimitations"] = "Votes are reported in terms of Agree minus Disagree."
    assert _hc().lint(doc) == []


@pytest.mark.parametrize("conforms", [CROISSANT_URI, [CROISSANT_URI], [RAI_URI], None])
def test_lint_wants_both_conforms_to_uris(conforms):
    doc = _good_doc()
    doc["conformsTo"] = conforms
    assert any("conformsTo" in p for p in _hc().lint(doc))


def test_overrides_use_the_prefixed_key():
    hc = _hc()
    merged, problems = hc.apply_overrides({"rai:dataBiases": "generated"}, {"rai:dataBiases": "edited"})
    assert merged == {"rai:dataBiases": "edited"} and problems == []
    _, problems = hc.apply_overrides({"rai:dataBiases": "generated"}, {"dataBiases": "edited"})
    assert problems and "rai:dataBiases" in problems[0]
    _, problems = hc.apply_overrides({}, ["not", "a", "dict"])
    assert problems


# ---------------------------------------------------------------------------------------------------------------------
# The rai override at staging, and the re-lint before --push-staged
# ---------------------------------------------------------------------------------------------------------------------

def _write_overrides(root, obj):
    with open(os.path.join(root, T, "card_overrides.json"), "w") as f:
        json.dump(obj, f)


def test_rai_override_replaces_a_generated_value(tmp_path):
    root, *_ = make_root(tmp_path)
    _write_overrides(root, {"rai": {"rai:dataBiases": "Edited bias text."}})
    _, d = pub.stage(root, T, lists=TEST_LISTS)
    assert _croissant(d)["rai:dataBiases"] == "Edited bias text."


@pytest.mark.parametrize("override, match", [
    ({"dataBiases": "x"}, "rai:dataBiases"),
    ({"rai:dataBiases": "Images are proprietary."}, "license field"),
    ({"rai:dataBiases": "TODO: later"}, "TODO"),
    ({"rai:dataCollectionTimeframe": "2020/2022"}, "unknown RAI property"),
])
def test_a_bad_rai_override_refuses_staging(tmp_path, override, match):
    root, *_ = make_root(tmp_path)
    _write_overrides(root, {"rai": override})
    with pytest.raises(pub.PublishError, match=match):
        pub.stage(root, T, lists=TEST_LISTS)


def test_verify_staged_relints_croissant(tmp_path, monkeypatch, capsys):
    root, d = _stage(tmp_path)
    assert pub.verify_staged(root, T, lists=LISTS_ALL_PUBLIC)[1] == d
    out = capsys.readouterr().out
    assert "croissant_validated" in out
    # a rule tightened after staging covers the old staging too
    monkeypatch.setattr(_hc(), "LICENCE_STEMS", _hc().LICENCE_STEMS + (r"panorama",))
    with pytest.raises(pub.PublishError, match="restage"):
        pub.verify_staged(root, T, lists=LISTS_ALL_PUBLIC)


LISTS_ALL_PUBLIC = dict(TEST_LISTS, mode="api")


# ---------------------------------------------------------------------------------------------------------------------
# Numbers come only from the staged files (private deployments)
# ---------------------------------------------------------------------------------------------------------------------

def _status_counts(manifest_path):
    return pd.read_csv(manifest_path)["status"].value_counts().to_dict()


def test_private_rows_never_reach_the_missing_data_counts(tmp_path):
    """The private rows carry a status public rows never have, at a count printed nowhere else: a count read from
    the root manifest (before the public filter) would show up here."""
    root, *_ = make_root(tmp_path)
    m_path = os.path.join(root, T, "manifest.csv")
    m = pd.read_csv(m_path)
    extra = pd.concat([m.iloc[[0]]] * 3).assign(city="zurich", label_uid=["zurich:5", "zurich:6", "zurich:7"],
                                               label_id=[5, 6, 7], status="no_depth",
                                               crop_path="correct/zurich_5.webp")
    pd.concat([m, extra]).to_csv(m_path, index=False)
    _, d = pub.stage(root, T, lists=LISTS)
    doc = _croissant(d)
    text = doc["rai:dataCollectionMissingData"]
    staged = _status_counts(os.path.join(d, "manifest.csv"))
    rooted = _status_counts(m_path)
    assert staged != rooted and "no_depth" in rooted and "no_depth" not in staged
    for status, n in staged.items():
        assert f"{status} {n}" in text
    assert "no_depth" not in text
    assert "zurich" not in json.dumps(doc).lower()


def test_every_number_in_the_rai_block_comes_from_a_staged_file_or_the_card(tmp_path):
    """Secondary check (small numbers appear on the card anyway); the direct assertion is the test above."""
    _, d = _stage(tmp_path)
    doc = _croissant(d)
    sources = [open(os.path.join(d, "dataset_info.json")).read(), open(os.path.join(d, "README.md")).read(),
               " ".join(f"{k} {v}" for k, v in _status_counts(os.path.join(d, "manifest.csv")).items()),
               json.dumps(_hc().vote_stats(pd.read_csv(os.path.join(d, "split.csv"))))]
    blob = "\n".join(sources)
    for v in _values(doc):
        for run in re.findall(r"\d+", v):
            assert run in blob, (run, v)


# ---------------------------------------------------------------------------------------------------------------------
# Card-derived text is the card's text
# ---------------------------------------------------------------------------------------------------------------------

def _sentences_without_todo(text):
    hc = _hc()
    return [s for s in hc.sentences(hc.plain_text(text)) if "TODO" not in s]


def _strip(text):
    return _hc().plain_text(text)


def test_card_derived_rai_text_appears_in_the_card(tmp_path):
    root, d = _stage(tmp_path)
    doc = _croissant(d)
    card = _strip(open(os.path.join(d, "README.md")).read())
    ds = _read_json(os.path.join(d, "dataset_info.json"))
    gen = _read_json(os.path.join(d, "generation_run.json"))
    prior = _read_json(os.path.join(d, "deployment_priors.json"))[T]
    fragments = {
        "rai:dataUseCases": _sentences_without_todo(hf_cards.intended_use_text(T))[:1],
        "rai:dataLimitations": _sentences_without_todo(hf_cards.intended_use_text(T))[1:],
        "rai:dataPreprocessingProtocol": (_sentences_without_todo(hf_cards.crop_text(ds, gen))
                                          + _sentences_without_todo(hf_cards.downsample_text(gen))
                                          + _sentences_without_todo(hf_cards.split_text(ds["heldout_cities"], None))),
        "rai:dataAnnotationProtocol": (_sentences_without_todo(hf_cards.VOTE_SOURCE_TEXT)
                                       + [_strip(rule) for _, rule in hf_cards.vote_rules(0)]),
        "rai:dataAnnotationAnalysis": _sentences_without_todo(hf_cards.prior_sentence(prior)),
        "rai:machineAnnotationTools": [_strip(hf_cards.ai_labels_text(False))],
    }
    for key, frags in fragments.items():
        assert frags, key
        for frag in frags:
            frag = frag.rstrip(".;")
            assert frag in doc[key] or frag[:1].upper() + frag[1:] in doc[key], (key, frag)
            assert frag in card or frag[:1].lower() + frag[1:] in card, (key, frag)
    # the versioning text, with the one phrase that points at the card's layout rewritten
    for frag in _sentences_without_todo(hf_cards.versioning_text(gen.get("selection_sha256"), None)):
        frag = frag.rstrip(".")
        assert frag in card
        rewritten = frag
        for old, new in _hc().CARD_TO_RAI_REWRITES.items():
            rewritten = rewritten.replace(old, new)
        assert rewritten in doc["rai:dataReleaseMaintenancePlan"]


# ---------------------------------------------------------------------------------------------------------------------
# Column descriptions
# ---------------------------------------------------------------------------------------------------------------------

PUBLISHABLE = [c for c in dg.MANIFEST_COLUMNS if c not in pub.PRIVATE_COLUMNS] + ["hard", "split"]


def test_every_publishable_column_is_described():
    assert set(PUBLISHABLE) - set(pub.CROISSANT_FIELDS) == set()
    for c, (dtype, desc) in pub.CROISSANT_FIELDS.items():
        assert dtype in ("sc:Text", "sc:Integer", "sc:Float"), c       # no sc:Boolean: it reads "f" and "no" as True
        assert desc.strip(), c
    for c in ("severity", "clipped", "hard"):
        assert pub.CROISSANT_FIELDS[c][0] == "sc:Integer"
    assert pub.CROISSANT_FIELDS["high_quality_user"][0] == "sc:Text"


def test_the_fixture_columns_are_all_described(tmp_path):
    _, d = _stage(tmp_path)
    columns = list(pd.read_csv(os.path.join(d, "split.csv"), nrows=0).columns)
    described = {f["name"] for f in _croissant(d)["recordSet"][0]["field"]}
    assert set(columns) == described


def test_an_undescribed_column_warns_and_is_recorded(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    split = pd.read_csv(os.path.join(root, T, "split.csv"), dtype=str)
    split["new_pipeline_column"] = "1"
    _rewrite_split(root, split)
    _, d = pub.stage(root, T, lists=TEST_LISTS)
    out = capsys.readouterr().out
    assert "WARNING" in out and "new_pipeline_column" in out
    rec = _read_json(os.path.join(root, T, "publish", pub.STAGING_RECORD))
    assert rec["croissant_undescribed_columns"] == ["new_pipeline_column"]
    pub.verify_staged(root, T, lists=LISTS_ALL_PUBLIC)
    assert "new_pipeline_column" in capsys.readouterr().out


def test_staging_record_says_whether_croissant_was_validated(tmp_path):
    root, _ = _stage(tmp_path)
    rec = _read_json(os.path.join(root, T, "publish", pub.STAGING_RECORD))
    assert isinstance(rec["croissant_validated"], bool)
    assert isinstance(rec["croissant_warnings"], list)
    try:
        import mlcroissant  # noqa: F401
    except ImportError:
        assert rec["croissant_validated"] is False
    else:
        assert rec["croissant_validated"] is True


# ---------------------------------------------------------------------------------------------------------------------
# With mlcroissant
# ---------------------------------------------------------------------------------------------------------------------

def test_mlcroissant_parses_every_rai_value_as_written(tmp_path):
    mlc = _require_mlcroissant()
    _, d = _stage(tmp_path)
    path = os.path.join(d, "croissant.json")
    doc = _read_json(path)
    assert set(MLCROISSANT_ATTRS) == set(_rai(doc))
    meta = mlc.Dataset(jsonld=path).metadata
    for key, attr in MLCROISSANT_ATTRS.items():
        got = getattr(meta, attr)
        want = doc[key]
        if isinstance(got, list) and not isinstance(want, list):
            want = [want]
        assert got == want, key


def test_mlcroissant_loads_every_column_with_blank_cells(tmp_path):
    mlc = _require_mlcroissant()
    root, *_ = make_root(tmp_path)
    split = pd.read_csv(os.path.join(root, T, "split.csv"), dtype=str, keep_default_na=False)
    for c in PUBLISHABLE:
        if c not in split.columns:
            split[c] = ""
    n = len(split)
    split["severity"] = ["3", ""] * (n // 2) + [""] * (n % 2)
    split["pose_pitch"] = ["1.5", ""] * (n // 2) + [""] * (n % 2)
    split["high_quality_user"] = ["True", "f"] * (n // 2) + [""] * (n % 2)
    split["tags"] = ["narrow;steep", ""] * (n // 2) + [""] * (n % 2)
    _rewrite_split(root, split)
    _, d = pub.stage(root, T, lists=TEST_LISTS)
    doc = _croissant(d)
    staged_columns = list(pd.read_csv(os.path.join(d, "split.csv"), nrows=0).columns)
    assert {f["name"] for f in doc["recordSet"][0]["field"]} == set(staged_columns) == set(PUBLISHABLE)
    ds = mlc.Dataset(jsonld=os.path.join(d, "croissant.json"), mapping={"split-csv": os.path.join(d, "split.csv")})
    records = list(ds.records("labels"))
    assert len(records) == n
    assert records[0]["labels/high_quality_user"] in (b"True", "True")


def test_validate_croissant_returns_the_warnings(tmp_path):
    _require_mlcroissant()
    _, d = _stage(tmp_path)
    validated, message, warnings = pub.validate_croissant(os.path.join(d, "croissant.json"))
    assert validated is True and isinstance(warnings, list)
    assert any("license" in w.lower() for w in warnings)      # no licence passed: Croissant 1.1 wants one


def test_require_mlcroissant_fails_in_ci_and_skips_elsewhere(monkeypatch):
    import sys

    from conftest import require_mlcroissant
    monkeypatch.setitem(sys.modules, "mlcroissant", None)          # import now raises ImportError
    monkeypatch.setenv("REQUIRE_MLCROISSANT", "1")
    with pytest.raises(pytest.fail.Exception, match="REQUIRE_MLCROISSANT"):
        require_mlcroissant()
    monkeypatch.delenv("REQUIRE_MLCROISSANT")
    with pytest.raises(pytest.skip.Exception):
        require_mlcroissant()
