"""fine_tune_classifier.py's --backbone / --learning-rate options (issue #119): which combinations the parser refuses.

``main`` parses and checks its arguments before importing anything heavy, so a combination that passes reaches
``import datasets``; the tests block that import (``sys.modules["datasets"] = None``) and read the ImportError as
"the parser accepted it"."""
import sys

import pytest

import fine_tune_classifier as ftc


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.delenv("SVA_BACKBONE_RANDOM_INIT", raising=False)
    split = tmp_path / "split.csv"
    split.write_text("label_uid,split\na:1,train\n")
    cities = tmp_path / "cities.json"
    cities.write_text('{"cities": []}')
    init = tmp_path / "init"
    init.mkdir()
    (init / "config.json").write_text("{}")

    def go(*argv):
        argv = [a.format(split=split, cities=cities, init=init) for a in argv]
        try:
            ftc.main(["--label_type", "CurbRamp", *argv])
        except ImportError:
            return "accepted"
        return "returned"
    return go


def refused(run, capsys, *argv, match):
    with pytest.raises(SystemExit) as e:
        run(*argv)
    assert e.value.code == 2
    assert match in capsys.readouterr().err


@pytest.mark.parametrize("other", [["--context-arm", "side_info"], ["--region-conditioning", "country",
                                                                     "--cities-json", "{cities}"],
                                   ["--init-from", "{init}"]])
def test_a_non_default_backbone_is_refused_with_another_experiment(run, capsys, other):
    refused(run, capsys, "--backbone", "convnextv2-base", "--output-dir", "o", *other,
            match="--backbone is compared on the plain model only")


@pytest.mark.parametrize("argv", [["--backbone", "dinov2-reg-large"], ["--learning-rate", "3e-5"]])
def test_a_non_default_backbone_or_rate_needs_an_output_dir(run, capsys, argv):
    refused(run, capsys, *argv, match="needs --output-dir")


def test_an_unknown_backbone_is_refused(run, capsys):
    refused(run, capsys, "--backbone", "resnet50", "--output-dir", "o", match="invalid choice")


def test_random_init_is_for_non_default_backbones_only(run, capsys, monkeypatch):
    monkeypatch.setenv("SVA_BACKBONE_RANDOM_INIT", "1")
    refused(run, capsys, "--output-dir", "o", match="SVA_BACKBONE_RANDOM_INIT=1 applies")
    assert run("--backbone", "dinov3-vitl16", "--output-dir", "o") == "accepted"


def test_allowed_combinations_parse(run):
    assert run("--backbone", "convnextv2-base", "--target", "soft", "--split", "{split}", "--output-dir", "y") == "accepted"
    assert run("--backbone", "rampnet-convnextv2-base", "--learning-rate", "3e-5", "--output-dir", "y") == "accepted"
    assert run("--backbone", "dinov2-large", "--context-arm", "side_info", "--output-dir", "y") == "accepted"
    assert run() == "accepted"   # the default run is unchanged
