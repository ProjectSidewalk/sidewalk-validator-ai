"""Golden renders of ``hf_cards.dataset_card``: the card text, byte for byte, for a small matrix of contexts.

The other card tests assert substrings, so a refactor of the card code that moves a space, a line break or a sentence
would pass them. These files pin the whole text. Synthetic cities only (``city-a``, ``city-b``, held-out ``city-c``)
and a fixed ``code_commit``.

A deliberate change to the card text updates the files: ``UPDATE_GOLDEN=1 pytest tests/test_dataset_card_golden.py``,
then review the diff of ``tests/fixtures/dataset_cards/`` like any other change.
"""
import copy
import os

import pytest

import hf_cards

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "dataset_cards")

BASE = {
    "label_type": "CurbRamp",
    "dataset_repo": "projectsidewalk/sidewalk-validator-ai-dataset-curbramp",
    "model_repo": "projectsidewalk/sidewalk-validator-ai-curbramp",
    "code_commit": "0123456789abcdef0123456789abcdef01234567",
    "dataset": {
        "n": 20, "splits": ["train", "val", "test", "test_heldout"],
        "counts": {"train": {"correct": 6, "incorrect": 4}, "val": {"correct": 2, "incorrect": 1},
                   "test": {"correct": 2, "incorrect": 2}, "test_heldout": {"correct": 1, "incorrect": 2}},
        "split_counts": {"train": 10, "val": 3, "test": 4, "test_heldout": 3},
        "class_totals": {"correct": 11, "incorrect": 9},
        "cities": {"city-a": 9, "city-b": 8, "city-c": 3}, "heldout_cities": ["city-c"],
        "capture_years": {"min": 2016, "max": 2023}, "n_panoramas": 14, "pano_sources": {"gsv": 20},
        "content_hash": "f" * 64,
    },
    "prior": {"correct": 900, "incorrect": 100, "rule": "majority2", "measured_on": "2026-10-17",
              "source": "rawLabels of city-a, city-b, city-c, human votes only",
              "rules": {"margin": {"correct": 700, "incorrect": 60}}},
    "generation_run": {"label_type": "CurbRamp", "seed": 42, "tilt_beta": 0.95, "depth_encoder": "vitb",
                       "selection_sha256": "abc123",
                       "settings": {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                                    "include_ai_labels": False}},
    "overrides": {"split_seed": 42},
}
LICENSE = "cc-by-4.0"


def _no_prior(ctx):
    ctx["prior"] = None


def _experts(ctx):
    ctx["generation_run"]["n_experts"] = 2
    ctx["generation_run"]["class_source_counts"] = {"margin": 18, "expert": 2}


def _non_gsv(ctx):
    ctx["dataset"]["pano_sources"] = {"gsv": 16, "mapillary": 3, "panoramax": 1}
    ctx["generation_run"]["settings"]["source_beta"] = {"mapillary": 0.0, "panoramax": 1.0}


def _no_heldout(ctx):
    ds = ctx["dataset"]
    ds["splits"] = ["train", "val", "test"]
    ds["counts"].pop("test_heldout")
    ds["split_counts"].pop("test_heldout")
    ds["heldout_cities"] = []


def _no_split_seed(ctx):
    ctx["overrides"].pop("split_seed")


def _sparse(ctx):
    """Every optional input missing at once: the card's TODO paths and the no-prior, no-expert, no-license text."""
    for f in (_no_prior, _non_gsv, _no_heldout, _no_split_seed):
        f(ctx)
    ctx["dataset"]["capture_years"] = {"min": None, "max": None}
    ctx["code_commit"] = None
    ctx["generation_run"] = {"settings": {"include_ai_labels": True}}


CASES = {  # name -> (edit of BASE, license id)
    "base": (None, LICENSE),
    "no_prior": (_no_prior, LICENSE),
    "experts": (_experts, LICENSE),
    "non_gsv": (_non_gsv, LICENSE),
    "no_heldout": (_no_heldout, LICENSE),
    "no_split_seed": (_no_split_seed, LICENSE),
    "no_license": (None, None),
    "experts_non_gsv_no_license": (lambda c: (_experts(c), _non_gsv(c)), None),
    "sparse": (_sparse, None),
}


def render(name):
    """The dataset card for case ``name``."""
    edit, license_id = CASES[name]
    ctx = copy.deepcopy(BASE)
    if edit:
        edit(ctx)
    return hf_cards.dataset_card(ctx, license_id)


@pytest.mark.parametrize("name", sorted(CASES))
def test_dataset_card_matches_its_golden_render(name):
    path = os.path.join(GOLDEN_DIR, f"{name}.md")
    text = render(name)
    if os.environ.get("UPDATE_GOLDEN") == "1":
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w", newline="") as f:
            f.write(text)
    with open(path, newline="") as f:
        assert text == f.read(), f"{name}: the card text changed; if on purpose, rerun with UPDATE_GOLDEN=1"


def test_the_matrix_covers_each_option():
    """Each varied input takes both values somewhere in the matrix."""
    cards = {name: render(name) for name in CASES}
    assert any("deployment prior" in c for c in cards.values()) and any("has no entry" in c for c in cards.values())
    assert any("Expert votes came from" in c for c in cards.values())
    assert any("Mapillary" in c for c in cards.values()) and any("Mapillary" not in c for c in cards.values())
    assert any("no cities were held out" in c for c in cards.values())
    assert any("(split seed 42)" in c for c in cards.values()) and any("split seed (split_data.py" in c
                                                                        for c in cards.values())
    assert any("**License:**" in c for c in cards.values()) and any("**License:**" not in c for c in cards.values())
