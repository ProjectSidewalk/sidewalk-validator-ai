"""publish_to_hf.py and hf_cards.py on a tiny fake pipeline output root (no GPU, no network, no token)."""
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import pytest

import evaluate
import hf_cards
import publish_to_hf as pub
from calibration import calibrate as cal

T = "CurbRamp"
# (city, label_id, class, split, capture date); label_id 7 repeats across cities on purpose
ROWS = [("teaneck", 7, "correct", "train", "2019-08"), ("teaneck", 8, "incorrect", "train", "2020-05"),
        ("chicago", 7, "correct", "train", "2021-06"), ("chicago", 9, "incorrect", "train", "2018-07"),
        ("teaneck", 10, "correct", "train", "2022-01"), ("teaneck", 11, "incorrect", "train", "2019-03"),
        ("teaneck", 12, "correct", "val", "2019-08"), ("chicago", 13, "incorrect", "val", "2017-09"),
        ("teaneck", 14, "correct", "test", "2023-04"), ("chicago", 15, "incorrect", "test", "2019-08"),
        ("oradell", 1, "correct", "test_heldout", "2016-10"), ("oradell", 2, "incorrect", "test_heldout", "2019-08")]


def _predictions(n, split, rng):
    """Synthetic predictions in evaluate.py's format, roughly 80% right."""
    true = np.where(np.arange(n) % 2 == 0, "correct", "incorrect")
    p_inc = np.clip(np.where(true == "incorrect", 0.75, 0.25) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return pd.DataFrame({"label_uid": [f"{split}:{i}" for i in range(n)], "split": split, "true": true,
                         "p_correct": 1 - p_inc, "p_incorrect": p_inc})


def make_root(tmp_path, label_type=T, rows=ROWS, max_steps=2, epochs=2):
    """A pipeline output root with crops, manifest, split, two checkpoints, metrics and a calibrated map."""
    root = tmp_path / "out"
    base = root / label_type
    records = []
    for i, (city, lid, cls, split, date) in enumerate(rows):
        crop = f"{cls}/{city}_{lid}.webp"
        (base / cls).mkdir(parents=True, exist_ok=True)
        (base / crop).write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 " + f"{label_type}:{city}:{lid}".encode())
        records.append({"label_uid": f"{city}:{lid}", "city": city, "label_id": lid, "pano_id": f"pano{i // 2}",
                        "label_type": label_type, "label_class": cls, "author": "human", "human_agree": 3,
                        "human_disagree": 0, "image_capture_date": date, "pano_source": "gsv",
                        "pano_path": f"/secret/store/{city}/pa/pano{i}.jpg", "depth_m": 5.0, "clipped": 0,
                        "crop_path": crop, "status": "ok", "split": split})
    manifest = pd.DataFrame(records).drop(columns="split")
    failed = manifest.iloc[[0]].assign(label_uid="teaneck:99", label_id=99, status="no_pano",
                                       crop_path="correct/teaneck_99.webp")
    pd.concat([manifest, failed]).to_csv(base / "manifest.csv", index=False)
    split = pd.DataFrame(records)
    split.insert(split.columns.get_loc("crop_path"), "hard", 0)
    split.to_csv(base / "split.csv", index=False)
    (base / "correct" / "stray_not_in_split.webp").write_bytes(b"not listed")
    (root / "deployment_priors.json").write_text(json.dumps({
        label_type: {"correct": 900, "incorrect": 100, "measured_on": "2026-10-17",
                     "source": "rawLabels of teaneck, chicago, human votes only"}}))
    (base / "generation_run.json").write_text(json.dumps({
        "label_type": label_type, "settings": {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                                               "include_ai_labels": False},
        "tilt_beta": 0.95, "depth_encoder": "vitb", "seed": 42, "pano_store": "/secret/store",
        "selection_sha256": "abc123"}))

    dinov2 = base / "dinov2"
    log = [{"epoch": 1.0, "eval_macro_f1": 0.81, "step": 2}, {"epoch": 2.0, "eval_macro_f1": 0.79, "step": 4}]
    for step in (2, 4):
        ck = dinov2 / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(f"weights-{step}".encode())
        (ck / "config.json").write_text(json.dumps({"architectures": ["Dinov2ForImageClassification"],
                                                    "id2label": {"0": "correct", "1": "incorrect"}}))
        (ck / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "BitImageProcessor"}))
        for junk in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "scaler.pt"):
            (ck / junk).write_bytes(b"training residue")
        (ck / "trainer_state.json").write_text(json.dumps({
            "best_metric": 0.81, "best_model_checkpoint": f"{label_type}/dinov2/checkpoint-2", "best_global_step": 2,
            "epoch": step / 2, "global_step": step, "max_steps": max_steps, "num_train_epochs": epochs,
            "train_batch_size": 8, "total_flos": 1e19, "stateful_callbacks": {"x": 1},
            "log_history": log[:step // 2]}))
    (dinov2 / "training_run.json").write_text(json.dumps({"label_type": label_type, "seed": 42, "epochs": epochs,
                                                         "base_checkpoint": "facebook/dinov2-large",
                                                         "best_model_checkpoint": "checkpoint-2",
                                                         "best_metric": "macro_f1", "best_metric_value": 0.81}))

    split_sha = hashlib.sha256((base / "split.csv").read_bytes()).hexdigest()
    (dinov2 / "run_key.json").write_text(json.dumps({"label_type": label_type, "split_sha256": split_sha}))

    rng = np.random.default_rng(0)
    # predictions.csv: exactly the split's evaluation rows (what check_bindings compares)
    ev = split[split["split"].isin(["val", "test", "test_heldout"])]
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                  "p_correct": 0.7, "p_incorrect": 0.3}).to_csv(base / "predictions.csv", index=False)
    # metrics and map from a larger synthetic set, so the numbers are not degenerate
    preds = pd.concat([_predictions(400, "val", rng), _predictions(200, "test", rng),
                       _predictions(100, "test_heldout", rng)])
    metrics = evaluate.report(preds)
    (base / "metrics.json").write_text(json.dumps(metrics))
    api_map, rep, _ = cal.calibrate(preds, {"correct": 0.9, "incorrect": 0.1})
    rep["label_type"] = label_type
    rep["prior_source"] = {"source": "rawLabels of teaneck, chicago, human votes only", "measured_on": "2026-10-17"}
    (base / "calibration").mkdir()
    (base / "calibration" / f"{label_type.lower()}.json").write_text(json.dumps(api_map, indent=4))
    (base / "calibration" / "calibration_report.json").write_text(json.dumps(rep, default=float))
    return str(root), metrics, api_map, rep


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may reach the Hub: requests.get fails and no token is ever picked up from the machine."""
    import requests

    def blocked(*a, **k):
        raise requests.ConnectionError("network disabled in tests")

    def no_api(required=True):
        if required:
            raise pub.PublishError("no token in tests")
        return None

    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(pub, "_hf_api", no_api)
    # stage() reads the city lists from the API unless given them; in tests every make_root() city is public and
    # a file still goes through the real parser
    monkeypatch.setattr(pub, "city_lists", lambda cities_api=None, public_cities_file=None:
                        TEST_LISTS if public_cities_file is None else
                        _real_city_lists(cities_api, public_cities_file))


_real_city_lists = pub.city_lists
TEST_LISTS = {"public": {r[0] for r in ROWS} | {"teaneck", "chicago"}, "non_public_terms": set(),
              "source": "tests", "read_at": "2026-10-22T00:00:00+00:00"}


@pytest.fixture
def staged(tmp_path):
    root, metrics, api_map, rep = make_root(tmp_path)
    model_dir, dataset_dir = pub.stage(root, T)
    return {"root": root, "model": model_dir, "dataset": dataset_dir, "metrics": metrics, "api_map": api_map,
            "report": rep}


# ---------------------------------------------------------------------------------------------------------------------
# Model staging
# ---------------------------------------------------------------------------------------------------------------------

def test_model_repo_holds_only_inference_files(staged):
    files = {p for p, _ in pub.list_files(staged["model"])}
    assert files == {"model.safetensors", "config.json", "preprocessor_config.json", "trainer_state.json",
                     "confidence_precision_map.json", "calibration_report.json", "metrics.json", "training_run.json",
                     "README.md"}
    # the best checkpoint (2), not the latest (4)
    with open(os.path.join(staged["model"], "model.safetensors"), "rb") as f:
        assert f.read() == b"weights-2"


def test_trainer_state_is_trimmed_with_full_history(staged):
    with open(os.path.join(staged["model"], "trainer_state.json")) as f:
        state = json.load(f)
    assert "stateful_callbacks" not in state and "total_flos" not in state
    assert state["best_model_checkpoint"] == "checkpoint-2"
    assert len(state["log_history"]) == 2       # from the latest checkpoint, so epochs after the best are kept


def test_map_is_copied_verbatim(staged):
    with open(os.path.join(staged["model"], "confidence_precision_map.json")) as f:
        assert json.load(f) == staged["api_map"]


@pytest.mark.parametrize("path", ["optimizer.pt", "scheduler.pt", "rng_state.pth", "rng_state_3.pth",
                                  "training_args.bin", "scaler.pt", "checkpoint-2060/config.json", "checkpoint-12"])
def test_forbidden_files(path):
    assert pub.forbidden_reason(path)


@pytest.mark.parametrize("path", ["model.safetensors", "config.json", "trainer_state.json", "README.md",
                                  "train/correct/teaneck_7.webp"])
def test_allowed_files(path):
    assert pub.forbidden_reason(path) is None


def test_a_training_file_in_the_staged_folder_blocks_upload(staged):
    with open(os.path.join(staged["model"], "optimizer.pt"), "wb") as f:
        f.write(b"x")
    with pytest.raises(pub.PublishError, match="optimizer.pt"):
        pub.assert_publishable(staged["model"])
    with pytest.raises(pub.PublishError, match="optimizer.pt"):
        pub.plan_push(staged["model"], "x/y", "model", lambda *a: None)


def test_copy_in_refuses_training_files(tmp_path):
    src = tmp_path / "optimizer.pt"
    src.write_bytes(b"x")
    with pytest.raises(pub.PublishError):
        pub._copy_in(str(src), str(tmp_path / "publish" / "model" / "optimizer.pt"), "copy")


def test_train_size_mismatch_only_warns(tmp_path, capsys):
    # 103 steps/epoch x 32 is about 3,300 images (the published CurbRamp model); the fake train split has 6
    root, *_ = make_root(tmp_path, max_steps=2575, epochs=25)
    pub.stage(root, T)
    assert "WARNING: train-size cross-check failed" in capsys.readouterr().out


def test_train_size_check_says_when_it_cannot_run():
    ok, msg = pub.check_train_size({"max_steps": 10}, 100)
    assert ok is None and "skipped" in msg


def test_batch_constants_match_fine_tune_classifier():
    """hf_cards copies the batch size and accumulation from fine_tune_classifier.py; they must stay equal."""
    with open(os.path.join(os.path.dirname(pub.__file__), "fine_tune_classifier.py")) as f:
        src = f.read()
    assert int(re.search(r"per_device_train_batch_size=(\d+)", src).group(1)) == hf_cards.TRAIN_BATCH_SIZE
    assert int(re.search(r"gradient_accumulation_steps=(\d+)", src).group(1)) == hf_cards.GRAD_ACCUM


# ---------------------------------------------------------------------------------------------------------------------
# Binding checks: weights, split, predictions, map
# ---------------------------------------------------------------------------------------------------------------------

def test_map_for_another_label_type_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "calibration", "calibration_report.json")
    report = json.loads(open(path).read())
    report["label_type"] = "Crosswalk"
    with open(path, "w") as f:
        json.dump(report, f)
    with pytest.raises(pub.PublishError, match="Crosswalk"):
        pub.stage(root, T)


def test_map_without_report_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    os.remove(os.path.join(root, T, "calibration", "calibration_report.json"))
    with pytest.raises(pub.PublishError, match="calibration_report"):
        pub.stage(root, T)


def test_split_changed_after_training_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    with open(os.path.join(root, T, "split.csv"), "a") as f:
        f.write("\n")
    with pytest.raises(pub.PublishError, match="changed after training"):
        pub.stage(root, T)
    os.remove(os.path.join(root, T, "dinov2", "run_key.json"))
    with pytest.raises(pub.PublishError, match="run_key.json"):
        pub.stage(root, T)


def test_predictions_for_another_split_are_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "predictions.csv")
    df = pd.read_csv(path)
    df.loc[0, "label_uid"] = "seattle:7"
    df.to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="1 extra, 1 missing"):
        pub.stage(root, T)
    os.remove(path)
    with pytest.raises(pub.PublishError, match="predictions.csv"):
        pub.stage(root, T)


def test_out_of_order_file_times_warn(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    weights = os.path.join(root, T, "dinov2", "checkpoint-2", "model.safetensors")
    later = os.path.getmtime(os.path.join(root, T, "predictions.csv")) + 100
    os.utime(weights, (later, later))
    pub.stage(root, T)
    assert "is newer than" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------------
# Dataset staging
# ---------------------------------------------------------------------------------------------------------------------

def test_dataset_layout_follows_split_csv(staged):
    crops = {p for p, _ in pub.list_files(staged["dataset"]) if p.endswith(".webp")}
    assert crops == {f"{s}/{c}/{city}_{lid}.webp" for city, lid, c, s, _ in ROWS}
    assert not any("stray_not_in_split" in p for p in crops)      # on disk but not in split.csv
    assert not any("teaneck_99" in p for p in crops)              # failed manifest row
    split = pd.read_csv(os.path.join(staged["dataset"], "split.csv"))
    assert "pano_path" not in split.columns
    assert all(os.path.exists(os.path.join(staged["dataset"], p)) for p in split.crop_path)
    manifest = pd.read_csv(os.path.join(staged["dataset"], "manifest.csv"))
    assert "pano_path" not in manifest.columns and len(manifest) == len(ROWS) + 1
    with open(os.path.join(staged["dataset"], "generation_run.json")) as f:
        assert "pano_store" not in json.load(f)
    with open(os.path.join(staged["dataset"], "deployment_priors.json")) as f:
        assert list(json.load(f)) == [T]


def test_crops_are_linked_or_copied_never_moved(staged):
    assert os.path.exists(os.path.join(staged["root"], T, "correct", "teaneck_7.webp"))


def test_copy_mode(tmp_path):
    root, *_ = make_root(tmp_path)
    _, dataset_dir = pub.stage(root, T, link_mode="copy")
    staged = os.path.join(dataset_dir, "train", "correct", "teaneck_7.webp")
    assert os.stat(staged).st_ino != os.stat(os.path.join(root, T, "correct", "teaneck_7.webp")).st_ino


def _content_hash(dataset_dir):
    with open(os.path.join(dataset_dir, "dataset_info.json")) as f:
        return json.load(f)["content_hash"]


def test_content_hash_is_stable_and_changes_with_a_crop(tmp_path):
    root, *_ = make_root(tmp_path)
    _, d = pub.stage(root, T)
    first = _content_hash(d)
    _, d = pub.stage(root, T)
    assert _content_hash(d) == first
    with open(os.path.join(d, "SHA256SUMS")) as f:
        lines = f.read().splitlines()
    assert len(lines) == len(ROWS) and lines == sorted(lines, key=lambda line: line.split("  ")[1])
    # change one crop's bytes: the hash must change
    with open(os.path.join(root, T, "incorrect", "chicago_9.webp"), "ab") as f:
        f.write(b"edited")
    _, d = pub.stage(root, T)
    assert _content_hash(d) != first


def test_label_type_mismatch_fails(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "manifest.csv")
    pd.read_csv(path).assign(label_type="Crosswalk").to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="label_type"):
        pub.stage(root, T)


def test_a_copied_dataset_is_refused(tmp_path):
    """The #7 bug: one label type's folder holding another's crops (here, relabelled to pass the column check)."""
    root, *_ = make_root(tmp_path)
    pub.stage(root, T)
    shutil.copytree(os.path.join(root, T), os.path.join(root, "Crosswalk"),
                    ignore=shutil.ignore_patterns("publish"))
    for name in ("manifest.csv", "split.csv"):
        path = os.path.join(root, "Crosswalk", name)
        pd.read_csv(path).assign(label_type="Crosswalk").to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="same content hash"):
        pub.stage_dataset(root, "Crosswalk", os.path.join(root, "Crosswalk", "publish", "dataset"))


def test_label_uid_must_be_city_and_id(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "split.csv")
    df = pd.read_csv(path)
    df.loc[0, "label_uid"] = "7"
    df.to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="label_uid"):
        pub.stage(root, T)


def test_missing_crop_fails(tmp_path):
    root, *_ = make_root(tmp_path)
    os.remove(os.path.join(root, T, "correct", "teaneck_14.webp"))
    with pytest.raises(pub.PublishError, match="do not exist"):
        pub.stage(root, T)


def test_croissant(staged):
    with open(os.path.join(staged["dataset"], "croissant.json")) as f:
        doc = json.load(f)
    assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.1"
    fields = {f["name"] for f in doc["recordSet"][0]["field"]}
    assert {"label_uid", "label_class", "split", "crop_path"} <= fields
    pytest.importorskip("mlcroissant")
    assert pub.validate_croissant(os.path.join(staged["dataset"], "croissant.json"))[0]


# ---------------------------------------------------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------------------------------------------------

def _read(path):
    with open(path) as f:
        return f.read()


def test_model_card_has_the_real_numbers(staged):
    card = _read(os.path.join(staged["model"], "README.md"))
    test = staged["metrics"]["test"]
    assert card.startswith("---\nlicense: mit\n")
    assert "base_model: facebook/dinov2-large" in card and "pipeline_tag: image-classification" in card
    assert "projectsidewalk/sidewalk-validator-ai-dataset-curbramp" in card
    assert f"{test['accuracy']:.3f}" in card and f"{test['auc']:.3f}" in card
    assert f"{test['balanced_accuracy']:.3f}" in card
    assert f"{test['per_class']['incorrect']['recall']:.3f}" in card
    assert "test_heldout" in card and "0.92" in card
    th = hf_cards.gate_thresholds(staged["api_map"])
    if th["correct"] is not None:
        assert f"{th['correct']:g}" in card
    assert f"{staged['report']['eval']['overall_ece']:.4f}" in card
    assert "checkpoint-2" in card
    assert "@inproceedings{saha2019projectsidewalk" in card
    assert "licens" not in card.lower().replace("license: mit", "")


def test_dataset_card_has_the_real_numbers(staged):
    card = _read(os.path.join(staged["dataset"], "README.md"))
    info = json.loads(_read(os.path.join(staged["dataset"], "dataset_info.json")))
    assert info["content_hash"] in card
    assert info["split_counts"] == {"train": 6, "val": 2, "test": 2, "test_heldout": 2}
    assert "2016–2023" in card                                  # capture years from image_capture_date
    assert "oradell" in card and "test_heldout" in card
    assert 'size_categories:\n  - "n<1K"' in card
    assert "TODO: dataset license" in card                     # not guessed
    assert "license:" not in card.split("---")[1].replace("# TODO: dataset license", "")
    assert "900" in card and "100" in card                      # deployment prior counts
    assert "disagree − agree > 1" in card and "agree − disagree > 2" in card


def test_dataset_card_states_the_expert_rule_only_when_used(tmp_path, staged):
    assert "expert" not in _read(os.path.join(staged["dataset"], "README.md")).lower()
    root, *_ = make_root(tmp_path / "with-experts")
    path = os.path.join(root, T, "generation_run.json")
    run = json.loads(_read(path))
    run.update(n_experts=2, class_source_counts={"margin": 9, "expert": 3})
    with open(path, "w") as f:
        json.dump(run, f)
    _, d = pub.stage(root, T)
    card = _read(os.path.join(d, "README.md"))
    assert "expert account voted for" in card and "2 project-team accounts" in card
    assert "9 margin and 3 expert" in card
    assert "met the `correct` margin rule" in card


def test_dataset_license_flag(tmp_path):
    root, *_ = make_root(tmp_path)
    _, d = pub.stage(root, T, dataset_license="cc-by-4.0")
    assert "\nlicense: cc-by-4.0\n" in _read(os.path.join(d, "README.md"))


def test_front_matter_is_valid_yaml(staged):
    yaml = pytest.importorskip("yaml")
    for folder in (staged["model"], staged["dataset"]):
        meta = yaml.safe_load(_read(os.path.join(folder, "README.md")).split("---")[1])
        assert "image-classification" in json.dumps(meta)
    meta = yaml.safe_load(_read(os.path.join(staged["model"], "README.md")).split("---")[1])
    results = meta["model-index"][0]["results"][0]["metrics"]
    assert {r["type"] for r in results} >= {"accuracy", "roc_auc"}


def test_yaml_quotes_ambiguous_strings():
    out = hf_cards.yaml_front_matter({"a": "1K<n<10K", "b": "yes", "c": "mit", "d": 0.5, "e": None})
    assert 'a: "1K<n<10K"' in out and 'b: "yes"' in out and "c: mit" in out and "d: 0.5" in out and "e: null" in out


def test_gate_thresholds():
    assert hf_cards.gate_thresholds({"correct": {"0.5": 0.9, "0.7": 0.92, "0.8": 0.95},
                                     "incorrect": {"0.5": 0.3}}) == {"correct": 0.7, "incorrect": None}


def test_yaml_floats_stay_floats():
    assert hf_cards._yaml_float(1e-05) == "1.0e-05" and hf_cards._yaml_float(3.0) == "3.0"
    assert hf_cards._yaml_float(float("inf")) == "null"
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(hf_cards.yaml_front_matter({"x": 1e-05}).split("---")[1])["x"] == 1e-05


def test_one_foreign_row_in_the_manifest_fails(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "manifest.csv")
    df = pd.read_csv(path)
    df.loc[3, "label_type"] = "Crosswalk"
    df.to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="Crosswalk"):
        pub.stage(root, T)


def test_card_configs_map_the_split_folders(staged):
    yaml = pytest.importorskip("yaml")
    meta = yaml.safe_load(_read(os.path.join(staged["dataset"], "README.md")).split("---")[1])
    files = {d["split"]: d["path"] for d in meta["configs"][0]["data_files"]}
    assert files == {"train": "train/*/*.webp", "validation": "val/*/*.webp", "test": "test/*/*.webp",
                     "test_heldout": "test_heldout/*/*.webp"}


def test_card_configs_text(staged):
    card = _read(os.path.join(staged["dataset"], "README.md"))
    assert '- split: validation\n        path: "val/*/*.webp"' in card
    assert '- split: test_heldout\n        path: "test_heldout/*/*.webp"' in card


def test_croissant_version_follows_the_content(staged):
    doc = json.loads(_read(os.path.join(staged["dataset"], "croissant.json")))
    info = json.loads(_read(os.path.join(staged["dataset"], "dataset_info.json")))
    assert doc["version"] == "1.0.0+" + info["content_hash"][:12]
    assert "datePublished" not in doc            # the fixture's generation_run has no start date


def test_marker_text_is_not_published(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "generation_run.json")
    run = json.loads(_read(path))
    run["pano_store_marker"] = "staged copy on /gscratch by jonf"
    with open(path, "w") as f:
        json.dump(run, f)
    _, d = pub.stage(root, T)
    assert "pano_store_marker" not in json.loads(_read(os.path.join(d, "generation_run.json")))


def test_symlinked_publish_dir_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("precious")
    os.symlink(target, os.path.join(root, T, "publish"))
    with pytest.raises(pub.PublishError, match="symlink"):
        pub.stage(root, T)
    assert (target / "keep.txt").exists()


def test_reset_dir_only_inside_publish(tmp_path):
    victim = tmp_path / "out" / "CurbRamp" / "data"
    victim.mkdir(parents=True)
    with pytest.raises(pub.PublishError, match="staging folder"):
        pub._reset_dir(str(victim), str(tmp_path / "out"))


# ---------------------------------------------------------------------------------------------------------------------
# Card overrides and the TODO gate
# ---------------------------------------------------------------------------------------------------------------------

FULL_OVERRIDES = {"failure_modes": ["Driveway aprons are accepted as curb ramps.", "Night panoramas."],
                  "split_seed": 7, "dataset_license": "cc-by-4.0", "doi": "10.57967/hf/1234"}


def test_overrides_fill_every_todo_and_the_dry_run_reports_zero(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    with open(os.path.join(root, T, "card_overrides.json"), "w") as f:
        json.dump(FULL_OVERRIDES, f)
    model_dir, dataset_dir = pub.stage(root, T)
    assert pub.count_todos(model_dir) == 0 and pub.count_todos(dataset_dir) == 0
    model_card, dataset_card = _read(os.path.join(model_dir, "README.md")), _read(os.path.join(dataset_dir, "README.md"))
    assert "- Driveway aprons are accepted as curb ramps.\n- Night panoramas.\n" in model_card
    assert "No DOI has been minted for this model yet." in model_card
    assert "(split seed 7)" in dataset_card and "doi.org/10.57967/hf/1234" in dataset_card
    assert "\nlicense: cc-by-4.0\n" in dataset_card
    capsys.readouterr()
    pub.push(T, model_dir, dataset_dir, public_lister=_lister(_fake_hub()), api=None)
    out = capsys.readouterr().out
    assert out.count("; 0 TODO markers in the card)") == 2


def test_doi_is_not_a_todo(staged):
    card = _read(os.path.join(staged["dataset"], "README.md"))
    assert "No DOI has been minted for this dataset yet." in card and "TODO: DOI" not in card


def test_unknown_override_key_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    with open(os.path.join(root, T, "card_overrides.json"), "w") as f:
        json.dump({"failure_mode": ["typo"]}, f)
    with pytest.raises(pub.PublishError, match="unknown keys"):
        pub.stage(root, T)


def test_push_staged_keeps_card_edits_and_detects_other_changes(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    model_dir, dataset_dir = pub.stage(root, T)
    card = os.path.join(dataset_dir, "README.md")
    edited = _read(card).replace("TODO:", "Filled by hand:")
    with open(card, "w") as f:
        f.write(edited)
    assert pub.verify_staged(root, T) == (model_dir, dataset_dir)
    assert pub.main(["--label_type", T, "--out", root, "--push-staged"]) == 0   # dry run; the offline listing is reported
    assert _read(card) == edited                                                       # not restaged
    with open(os.path.join(model_dir, "metrics.json"), "a") as f:
        f.write(" ")
    with pytest.raises(pub.PublishError, match="metrics.json"):
        pub.verify_staged(root, T)


def test_push_staged_detects_a_changed_crop_and_a_retrain(tmp_path):
    root, *_ = make_root(tmp_path)
    _, dataset_dir = pub.stage(root, T)
    crop = os.path.join(dataset_dir, "train", "correct", "teaneck_7.webp")
    os.remove(crop)                                  # break the hard link, then change the bytes
    with open(crop, "wb") as f:
        f.write(b"other")
    with pytest.raises(pub.PublishError, match="crops changed"):
        pub.verify_staged(root, T)
    pub.stage(root, T, link_mode="copy")       # with hard links the staged weights would change too (also refused)
    with open(os.path.join(root, T, "dinov2", "checkpoint-2", "model.safetensors"), "ab") as f:
        f.write(b"retrained")
    with pytest.raises(pub.PublishError, match="best checkpoint changed"):
        pub.verify_staged(root, T)


def test_push_staged_needs_a_staging_record(tmp_path):
    root, *_ = make_root(tmp_path)
    with pytest.raises(pub.PublishError, match="stage first"):
        pub.verify_staged(root, T)


def _mark_fixed_crop(root, record):
    """Record ``fixed_crop_half_px`` the way data_generation.py (``generation_run.json``, top level and under
    ``settings``) or fine_tune_classifier.py (``dinov2/training_run.json``, top level) writes it."""
    path = os.path.join(root, T, record)
    run = json.loads(_read(path))
    run["fixed_crop_half_px"] = 300
    if "settings" in run:
        run["settings"]["fixed_crop_half_px"] = 300
    with open(path, "w") as f:
        json.dump(run, f)


@pytest.mark.parametrize("record", ["generation_run.json", os.path.join("dinov2", "training_run.json")])
def test_a_fixed_crop_ablation_is_never_staged(tmp_path, capsys, record):
    root, *_ = make_root(tmp_path)
    _mark_fixed_crop(root, record)
    with pytest.raises(pub.PublishError, match="serving sizes each crop by depth.*ablation-only"):
        pub.stage(root, T)
    assert not os.path.exists(os.path.join(root, T, "publish"))  # refused before anything is staged
    assert pub.main(["--label_type", T, "--out", root]) == pub.EXIT_ERROR
    assert "--fixed-crop-half-px" in capsys.readouterr().err


def test_push_staged_refuses_a_model_retrained_as_the_ablation(tmp_path):
    root, *_ = make_root(tmp_path)
    pub.stage(root, T)
    _mark_fixed_crop(root, os.path.join("dinov2", "training_run.json"))
    with pytest.raises(pub.PublishError, match="ablation-only"):
        pub.verify_staged(root, T)


def test_cards_refuse_fixed_crop_runs(staged):
    ctx = {"label_type": T, "model_repo": "m", "dataset_repo": "d", "dataset": {}, "code_commit": None,
           "overrides": {}, "training_run": {"fixed_crop_half_px": 300}}
    with pytest.raises(ValueError, match="fixed half-width of 300 px"):
        hf_cards.model_card(ctx)
    with pytest.raises(ValueError, match="fixed half-width of 300 px"):
        hf_cards.dataset_card({**ctx, "generation_run": {"settings": {"fixed_crop_half_px": 300}}})
    model_run = json.loads(_read(os.path.join(staged["model"], "training_run.json")))
    gen_run = json.loads(_read(os.path.join(staged["dataset"], "generation_run.json")))
    assert hf_cards.fixed_crop_half_px(model_run, gen_run) is None  # the depth-sized fixture publishes


# ---------------------------------------------------------------------------------------------------------------------
# Audit and listings (mocked API, no network)
# ---------------------------------------------------------------------------------------------------------------------

def _fake_hub():
    """Listings shaped like the live repos in October 2026: curbramp's dataset is crosswalk's."""
    model = [{"path": p, "size": s, "blob": p, "sha256": None} for p, s in
             ((".gitattributes", 1519), ("README.md", 24), ("config.json", 1272),
              ("model.safetensors", 1_217_542_512), ("optimizer.pt", 2_435_342_411),
              ("preprocessor_config.json", 436), ("rng_state.pth", 14_645), ("scheduler.pt", 1_465),
              ("trainer_state.json", 40_000), ("training_args.bin", 5_368))]
    crosswalk = [{"path": p, "size": 100, "blob": "b" + p, "sha256": "s" + p} for p in
                 ("train/correct/chicago_1.webp", "train/incorrect/chicago_2.webp", "test/correct/chicago_10689.webp")]
    listings = {("dataset", "projectsidewalk/sidewalk-validator-ai-dataset-crosswalk"): crosswalk,
                # a near-copy: crosswalk's crops plus one more, so the sets are not identical
                ("dataset", "projectsidewalk/sidewalk-validator-ai-dataset-curbramp"):
                    crosswalk + [{"path": ".gitattributes", "size": 1, "blob": "g", "sha256": None},
                                 {"path": "val/correct/teaneck_5.webp", "size": 1, "blob": "x", "sha256": "sx"}]}
    for t in pub.LABEL_TYPES:
        listings[("model", pub.model_repo_id(t))] = model if t != "Crosswalk" else [
            f for f in model if not pub.forbidden_reason(f["path"])]
        listings.setdefault(("dataset", pub.dataset_repo_id(t)),
                            [{"path": f"train/correct/{t}_{i}.webp", "size": 1, "blob": f"{t}{i}", "sha256": f"h{t}{i}"}
                             for i in range(3)])
    del listings[("dataset", pub.dataset_repo_id("Obstacle"))]   # a missing repo
    return lambda repo_type, repo_id: listings.get((repo_type, repo_id))


def _lister(fetch, sha="a" * 40):
    """A listing function (``{"sha", "files"}``) over a ``_fake_hub`` file fetcher."""
    def lister(repo_type, repo_id):
        files = fetch(repo_type, repo_id)
        return None if files is None else {"sha": sha, "files": files}
    return lister


def test_audit_finds_stray_files_and_near_copies():
    result = pub.audit(fetch=_fake_hub())
    stray = {r["repo"]: [p for p, _ in r["stray"]] for r in result["models"]}
    assert sorted(stray["projectsidewalk/sidewalk-validator-ai-curbramp"]) == [
        "optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"]
    assert stray["projectsidewalk/sidewalk-validator-ai-crosswalk"] == []
    assert result["duplicates"] == [("projectsidewalk/sidewalk-validator-ai-dataset-curbramp",
                                     "projectsidewalk/sidewalk-validator-ai-dataset-crosswalk", 1.0, 1.0)]
    text = pub.format_audit(result)
    assert "optimizer.pt (2.3 GB)" in text
    assert "missing or private" in text
    assert "100% of names, 100% of content" in text
    assert "stub (24 B)" in text                                  # the live repos' license-only README
    assert "16 stray training files" in text and "1 pair(s)" in text


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def test_fetch_repo_listing_parses_the_api():
    calls = []

    def get(url, params=None, timeout=None):
        calls.append((url, params))
        if "missing" in url:
            return _Resp(401, {})
        return _Resp(200, {"sha": "abc", "siblings": [
            {"rfilename": "optimizer.pt", "size": 5, "blobId": "x"},
            {"rfilename": "train/correct/a_1.webp", "size": 9, "blobId": "y", "lfs": {"sha256": "deadbeef"}}]})

    assert pub.fetch_repo_listing("model", "org/a", get=get) == {"sha": "abc", "files": [
        {"path": "optimizer.pt", "size": 5, "blob": "x", "sha256": None},
        {"path": "train/correct/a_1.webp", "size": 9, "blob": "y", "sha256": "deadbeef"}]}
    assert pub.fetch_repo_files("dataset", "org/missing", get=get) is None
    assert calls[0] == ("https://huggingface.co/api/models/org/a", {"blobs": "true"})
    assert calls[1][0] == "https://huggingface.co/api/datasets/org/missing"


def test_network_errors_become_remote_errors(monkeypatch, capsys):
    import requests

    def down(*a, **k):
        raise requests.ConnectionError("no route to host")

    with pytest.raises(pub.RemoteError, match="no route"):
        pub.fetch_repo_listing("model", "org/a", get=down)
    monkeypatch.setattr(requests, "get", down)
    assert pub.main(["--audit"]) == pub.EXIT_REMOTE
    assert "Hub failure" in capsys.readouterr().err


def test_overlap_fraction():
    assert pub.overlap_fraction({1, 2, 3}, {1, 2, 3, 4}) == 1.0
    assert pub.overlap_fraction({1, 2}, {3}) == 0.0
    assert pub.overlap_fraction(set(), {1}) == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# Push and prune (fake HfApi; huggingface_hub not needed)
# ---------------------------------------------------------------------------------------------------------------------

class _Add:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo, self.path_or_fileobj = path_in_repo, path_or_fileobj


class _Delete:
    def __init__(self, path_in_repo):
        self.path_in_repo = path_in_repo


class _File:
    def __init__(self, path, size=1, sha256=None):
        self.path, self.size, self.blob_id = path, size, "b"
        self.lfs = {"sha256": sha256} if sha256 else None


class FakeApi:
    """Records calls; serves listings from a dict ``{(repo_type, repo_id): [paths]}`` (absent = no such repo)."""

    def __init__(self, repos=None, fail_listing=False):
        self.calls, self.repos, self.fail_listing = [], dict(repos or {}), fail_listing

    def repo_info(self, repo_id, repo_type=None):
        if self.fail_listing:
            import requests
            raise requests.ConnectionError("503 from the Hub")
        if (repo_type, repo_id) not in self.repos:
            raise LookupError(repo_id)          # authenticated_listing treats this as "no such repo" in tests
        return type("Info", (), {"sha": "f" * 40})()

    def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=False):
        return [_File(p) for p in self.repos[(repo_type, repo_id)]]

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))
        self.repos[(kw["repo_type"], repo_id)] = [".gitattributes"]

    fail_commit_for = None
    """repo_id whose create_commit raises (after recording the attempt), to simulate a Hub rejection."""

    def create_commit(self, **kw):
        self.calls.append(("create_commit", kw))
        if kw["repo_id"] == self.fail_commit_for:
            raise RuntimeError("412 Precondition Failed")
        return "commit-url"

    def delete_files(self, **kw):
        self.calls.append(("delete_files", kw))

    def upload_folder(self, **kw):              # must never be used: it can split a push into many commits
        raise AssertionError("upload_folder called")


@pytest.fixture
def fake_ops(monkeypatch):
    monkeypatch.setattr(pub, "_operation_classes", lambda: (_Add, _Delete))
    try:
        from huggingface_hub.errors import RepositoryNotFoundError  # noqa: F401
    except ImportError:
        return
    # with huggingface_hub installed, authenticated_listing catches its RepositoryNotFoundError, so map ours to it
    import huggingface_hub.errors as errors
    monkeypatch.setattr(errors, "RepositoryNotFoundError", LookupError)


def _official(t=T):
    return {("model", pub.model_repo_id(t)): [".gitattributes", "README.md", "model.safetensors", "optimizer.pt"],
            ("dataset", pub.dataset_repo_id(t)): [".gitattributes", "train/correct/chicago_1.webp"],
            **{("dataset", pub.dataset_repo_id(o)): [".gitattributes"] for o in pub.LABEL_TYPES if o != t}}


def test_dry_run_reports_and_never_commits(staged, capsys, fake_ops):
    api = FakeApi(_official())
    pub.push(T, staged["model"], staged["dataset"], api=api)
    assert [c for c in api.calls if c[0] != "create_repo"] == [] and not api.calls
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "files this commit deletes: 1: optimizer.pt" in out
    assert "TODO markers in the card" in out


def test_push_yes_is_exactly_one_commit_per_repo(tmp_path, fake_ops):
    root, *_ = make_root(tmp_path)
    with open(os.path.join(root, T, "card_overrides.json"), "w") as f:
        json.dump(FULL_OVERRIDES, f)
    model_dir, dataset_dir = pub.stage(root, T)
    api = FakeApi(_official())
    pub.push(T, model_dir, dataset_dir, yes=True, api=api)
    commits = [c[1] for c in api.calls if c[0] == "create_commit"]
    assert [c["repo_id"] for c in commits] == [pub.dataset_repo_id(T), pub.model_repo_id(T)]
    assert all(c["parent_commit"] == "f" * 40 for c in commits)
    model_ops = commits[1]["operations"]
    assert {o.path_in_repo for o in model_ops if isinstance(o, _Delete)} == {"optimizer.pt"}
    assert {o.path_in_repo for o in model_ops if isinstance(o, _Add)} == {p for p, _ in pub.list_files(model_dir)}
    dataset_ops = commits[0]["operations"]
    assert {o.path_in_repo for o in dataset_ops if isinstance(o, _Delete)} == {"train/correct/chicago_1.webp"}
    assert not any(c[0] == "create_repo" for c in api.calls)


def test_push_yes_refuses_todo_cards(staged, fake_ops):
    api = FakeApi(_official())
    with pytest.raises(pub.PublishError, match="TODO"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, api=api)
    assert not api.calls
    pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert sum(c[0] == "create_commit" for c in api.calls) == 2


def test_checks_for_both_repos_run_before_any_upload(staged, fake_ops):
    """N2: a problem in the model folder stops the dataset upload too."""
    with open(os.path.join(staged["model"], "scaler.pt"), "wb") as f:
        f.write(b"x")
    api = FakeApi(_official())
    with pytest.raises(pub.PublishError, match="scaler.pt"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert not api.calls


def test_push_yes_refuses_when_the_remote_cannot_be_listed(staged, fake_ops, capsys):
    api = FakeApi(_official(), fail_listing=True)
    pub.push(T, staged["model"], staged["dataset"], api=api)          # the dry run reports it
    assert "could not list" in capsys.readouterr().out
    with pytest.raises(pub.PublishError, match="could not"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert not api.calls


def test_push_refuses_crops_found_in_another_type(staged, fake_ops, capsys):
    """S3: the staged crops already sit in the Crosswalk dataset repo (the #7 bug, caught remotely)."""
    sums = pub.read_sha256sums(staged["dataset"])
    repos = _official()
    repos[("dataset", pub.dataset_repo_id("Crosswalk"))] = list(sums)        # same paths ...
    api = FakeApi(repos)
    api.list_repo_tree = lambda repo_id, repo_type=None, revision=None, recursive=False: [
        _File(p, sha256=sums.get(p)) for p in repos[(repo_type, repo_id)]]   # ... and same content
    pub.push(T, staged["model"], staged["dataset"], api=api)
    assert "100.0% of the staged crops" in capsys.readouterr().out
    with pytest.raises(pub.PublishError, match="already in projectsidewalk/sidewalk-validator-ai-dataset-crosswalk"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert not [c for c in api.calls if c[0] == "create_commit"]


def test_missing_repo_under_the_default_prefix_is_never_created(staged, fake_ops):
    repos = _official()
    del repos[("model", pub.model_repo_id(T))]
    api = FakeApi(repos)
    with pytest.raises(pub.PublishError, match="never created"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, create_repo=True, api=api)
    assert not api.calls


def test_test_prefix_needs_create_repo_and_creates_private(staged, fake_ops):
    prefix = "jonfroehlich/test-validator"
    api = FakeApi({})
    with pytest.raises(pub.PublishError, match="--create-repo"):
        pub.push(T, staged["model"], staged["dataset"], prefix=prefix, yes=True, allow_todo=True, api=api)
    pub.push(T, staged["model"], staged["dataset"], prefix=prefix, yes=True, allow_todo=True, create_repo=True,
             api=api)
    creates = [c for c in api.calls if c[0] == "create_repo"]
    assert len(creates) == 2 and all(c[2]["private"] is True and c[2]["exist_ok"] is False for c in creates)
    assert sum(c[0] == "create_commit" for c in api.calls) == 2


@pytest.mark.parametrize("prefix", ["projectsidewalk/sidewalk-validator-ai-", "sidewalk-validator-ai", "a/b/c",
                                    "", "org/"])
def test_bad_prefixes_are_refused(prefix):
    with pytest.raises(pub.PublishError, match="repo-prefix"):
        pub.validate_prefix(prefix)


def test_good_prefixes():
    pub.validate_prefix(pub.DEFAULT_PREFIX)
    pub.validate_prefix("jonfroehlich/test.validator_2")


def test_prune_remote(capsys):
    fetch = _lister(_fake_hub(), sha="c" * 40)
    api = FakeApi()
    assert pub.prune_remote("projectsidewalk/sidewalk-validator-ai-curbramp", fetch=fetch, api=api)
    assert api.calls == []                                       # dry run
    stray = pub.prune_remote("projectsidewalk/sidewalk-validator-ai-curbramp", yes=True, fetch=fetch, api=api)
    call = api.calls[0][1]
    assert call["delete_patterns"] == stray and "model.safetensors" not in stray and call["parent_commit"] == "c" * 40
    assert pub.prune_remote("projectsidewalk/sidewalk-validator-ai-crosswalk", yes=True, fetch=fetch, api=api) == []


def test_repo_prefix():
    assert pub.model_repo_id("NoCurbRamp", "me/test") == "me/test-nocurbramp"
    assert pub.dataset_repo_id("NoCurbRamp") == "projectsidewalk/sidewalk-validator-ai-dataset-nocurbramp"


def test_cli_stage(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    assert pub.main(["--label_type", T, "--out", root]) == 0
    assert "TODO markers to fill before --yes" in capsys.readouterr().out
    assert pub.main(["--label_type", "all", "--out", root]) == 1
    assert pub.main(["--label_type", T, "--out", root, "--repo-prefix", "oops"]) == 1


# ---------------------------------------------------------------------------------------------------------------------
# Round-2 review: partial failures, uncomparable repos, untested guards, nits
# ---------------------------------------------------------------------------------------------------------------------

def _filled_root(tmp_path):
    root, *_ = make_root(tmp_path)
    with open(os.path.join(root, T, "card_overrides.json"), "w") as f:
        json.dump(FULL_OVERRIDES, f)
    return root


def test_a_failed_model_commit_names_what_landed_and_how_to_retry(tmp_path, fake_ops, monkeypatch, capsys):
    root = _filled_root(tmp_path)
    pub.stage(root, T)
    api = FakeApi(_official())
    api.fail_commit_for = pub.model_repo_id(T)
    monkeypatch.setattr(pub, "_hf_api", lambda required=True: api)
    assert pub.main(["--label_type", T, "--out", root, "--push-staged", "--yes"]) == pub.EXIT_REMOTE
    err = capsys.readouterr().err
    assert f"Already committed in this run: {pub.dataset_repo_id(T)}" in err
    assert "--push-staged --what model --yes" in err and f"--out {os.path.abspath(root)}" in err
    # the retry the message prints commits the model, and only the model, in one commit
    api.fail_commit_for = None
    api.calls.clear()
    assert pub.main(["--label_type", T, "--out", root, "--push-staged", "--what", "model", "--yes"]) == 0
    commits = [c[1]["repo_id"] for c in api.calls if c[0] == "create_commit"]
    assert commits == [pub.model_repo_id(T)]


def test_a_failed_first_commit_says_nothing_landed(staged, fake_ops):
    api = FakeApi(_official())
    api.fail_commit_for = pub.dataset_repo_id(T)
    with pytest.raises(pub.RemoteError, match="Already committed in this run: none.*--what both"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)


def test_a_failed_prune_is_a_remote_error():
    class Broken(FakeApi):
        def delete_files(self, **kw):
            raise RuntimeError("401")
    with pytest.raises(pub.RemoteError, match="nothing was deleted"):
        pub.prune_remote("projectsidewalk/sidewalk-validator-ai-curbramp", yes=True,
                         fetch=_lister(_fake_hub()), api=Broken())


def test_crops_without_sha256_cannot_be_compared(staged, fake_ops, capsys):
    """F2: another repo's crops stored without LFS have no sha256; that must not read as 0% overlap."""
    repos = _official()
    repos[("dataset", pub.dataset_repo_id("Crosswalk"))] = list(pub.read_sha256sums(staged["dataset"]))
    api = FakeApi(repos)                       # _File without sha256: non-LFS
    pub.push(T, staged["model"], staged["dataset"], api=api)
    assert "WARNING: could not compare crops with projectsidewalk/sidewalk-validator-ai-dataset-crosswalk" in \
        capsys.readouterr().out
    with pytest.raises(pub.PublishError, match="could not compare crops"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert not api.calls


def test_a_listing_error_for_another_dataset_repo_is_refused(staged, fake_ops):
    api = FakeApi(_official())
    real = api.repo_info

    def flaky(repo_id, repo_type=None):
        if repo_id == pub.dataset_repo_id("Obstacle"):
            import requests
            raise requests.Timeout("read timed out")
        return real(repo_id, repo_type=repo_type)
    api.repo_info = flaky
    with pytest.raises(pub.PublishError, match="could not compare with the other dataset repos"):
        pub.push(T, staged["model"], staged["dataset"], yes=True, allow_todo=True, api=api)
    assert not api.calls


def test_validation_folder_counts_as_crops_in_remote_listings():
    files = [{"path": "validation/correct/a_1.webp", "sha256": "x"}, {"path": "val/correct/a_2.webp", "sha256": "y"}]
    assert pub.crop_hashes(files) == {"x", "y"}
    assert not pub._is_crop_path("validation/correct/a_1.webp")      # staged layout uses val/


def test_push_staged_refuses_a_card_without_the_hash_line(tmp_path):
    root, *_ = make_root(tmp_path)
    _, dataset_dir = pub.stage(root, T)
    card = os.path.join(dataset_dir, "README.md")
    text = _read(card)
    with open(card, "w") as f:                 # the hash still appears elsewhere, but not on its own line
        f.write(re.sub(r"- \*\*Content hash:\*\* `([0-9a-f]+)`", r"- **Content hash:** see SHA256SUMS\n\nOld: `\1`",
                       text))
    assert json.loads(_read(os.path.join(dataset_dir, "dataset_info.json")))["content_hash"] in _read(card)
    with pytest.raises(pub.PublishError, match="Content hash"):
        pub.verify_staged(root, T)


def test_push_staged_refuses_a_crop_deleted_after_staging(tmp_path):
    root, *_ = make_root(tmp_path)
    _, dataset_dir = pub.stage(root, T)
    os.remove(os.path.join(dataset_dir, "test", "correct", "teaneck_14.webp"))
    with pytest.raises(pub.PublishError, match="differ from SHA256SUMS"):
        pub.verify_staged(root, T)


def test_push_staged_refuses_another_prefix(tmp_path):
    root, *_ = make_root(tmp_path)
    pub.stage(root, T)
    with pytest.raises(pub.PublishError, match="restage"):
        pub.verify_staged(root, T, prefix="jonfroehlich/test-validator")


def test_report_without_label_type_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "calibration", "calibration_report.json")
    report = json.loads(_read(path))
    del report["label_type"]
    with open(path, "w") as f:
        json.dump(report, f)
    with pytest.raises(pub.PublishError, match="is for None"):
        pub.stage(root, T)


def test_run_key_for_another_label_type_is_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "dinov2", "run_key.json")
    key = json.loads(_read(path))
    key["label_type"] = "Crosswalk"
    with open(path, "w") as f:
        json.dump(key, f)
    with pytest.raises(pub.PublishError, match="run_key.json is for 'Crosswalk'"):
        pub.stage(root, T)


def test_duplicate_prediction_rows_are_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    path = os.path.join(root, T, "predictions.csv")
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(pub.PublishError, match="duplicate"):
        pub.stage(root, T)


@pytest.mark.parametrize("prefix", ["org/name\n", "../x", "a/..x", "org/a--b"])
def test_more_bad_prefixes(prefix):
    with pytest.raises(pub.PublishError, match="repo-prefix"):
        pub.validate_prefix(prefix)


def test_org_namespace_in_any_case_is_never_created(staged, fake_ops):
    prefix = "ProjectSidewalk/sidewalk-validator-ai-test"
    assert pub.is_org_namespace(prefix)
    api = FakeApi({})
    with pytest.raises(pub.PublishError, match="never created"):
        pub.push(T, staged["model"], staged["dataset"], prefix=prefix, yes=True, allow_todo=True, create_repo=True,
                 api=api)
    assert not api.calls


def test_model_card_says_when_a_verdict_is_held_back(staged):
    """A held-back verdict is stated on the card with the cap, the reason and what the uncapped map would have done."""
    rep = json.loads(json.dumps(staged["report"]))
    rep["hold_back"] = {"verdicts": ["incorrect"], "cap": 0.9199, "reason": "prior under review (issue #56)",
                        "uncapped_reaches_gate": {"correct": True, "incorrect": True},
                        "uncapped_eval": {"incorrect": {"gate_coverage": 0.42, "gate_precision": 0.97, "gate_n": 98}}}
    capped = {v: ({k: min(x, 0.9199) for k, x in m.items()} if v == "incorrect" else m)
              for v, m in staged["api_map"].items()}
    section = hf_cards._calibration_section({"label_type": T, "api_map": capped, "calibration_report": rep})
    assert "**Held back:** the `incorrect` map is capped at 0.9199" in section and "serves 0.9199" in section
    assert "prior under review (issue [#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56))" in section
    assert "coverage 0.42" in section and "precision 0.97" in section and "n 98" in section
    assert "| incorrect | never reaches the gate |" in section
    # when the uncapped map never reached the gate, the card says the cap is a no-op instead of quoting 0.000 / n/a
    rep["hold_back"]["uncapped_reaches_gate"]["incorrect"] = False
    rep["hold_back"]["uncapped_eval"] = {"incorrect": {"gate_coverage": 0.0, "gate_precision": None, "gate_n": 0}}
    noop = hf_cards._calibration_section({"label_type": T, "api_map": capped, "calibration_report": rep})
    assert "the cap changes nothing: the uncapped map never reached the gate either" in noop
    assert "coverage 0" not in noop
    plain = hf_cards._calibration_section({"label_type": T, "api_map": staged["api_map"],
                                           "calibration_report": staged["report"]})
    assert "Held back" not in plain


# ---------------------------------------------------------------------------------------------------------------------
# Model-only staging (--what model): a machine with the checkpoints and the split, but without the crops
# ---------------------------------------------------------------------------------------------------------------------

def _remove_crops(root, label_type=T):
    """Delete every crop under <root>/<label_type>, as on a machine that only pulled the checkpoints."""
    import shutil
    for cls in pub.CLASSES:
        shutil.rmtree(os.path.join(root, label_type, cls), ignore_errors=True)


def test_stage_model_only_needs_no_crops(tmp_path, capsys):
    root, _, api_map, _ = make_root(tmp_path)
    _remove_crops(root)
    with pytest.raises(pub.PublishError, match="crops listed in .* do not exist"):
        pub.stage(root, T)                                   # a full staging still refuses
    model_dir, dataset_dir = pub.stage(root, T, what="model")
    assert dataset_dir is None
    staged = {rel for rel, _ in pub.list_files(model_dir)}
    assert staged == {"README.md", "calibration_report.json", "config.json", "confidence_precision_map.json",
                      "metrics.json", "model.safetensors", "preprocessor_config.json", "trainer_state.json",
                      "training_run.json"}
    assert json.load(open(os.path.join(model_dir, "confidence_precision_map.json"))) == api_map
    record = json.load(open(os.path.join(root, T, "publish", pub.STAGING_RECORD)))
    assert record["what"] == "model" and record["content_hash"] is None
    assert all(k.startswith("model/") for k in record["files"])
    card = open(os.path.join(model_dir, "README.md")).read()
    assert "TODO: dataset not staged with this model" in card and "`None`" not in card
    assert "train 6" in card                                 # split sizes still come from split.csv


def test_push_staged_model_only_needs_what_model(tmp_path):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model")
    with pytest.raises(pub.PublishError, match="model-only staging; pass --what model"):
        pub.verify_staged(root, T)
    with pytest.raises(pub.PublishError, match="model-only staging; pass --what model"):
        pub.verify_staged(root, T, what="dataset")
    assert pub.verify_staged(root, T, what="model") == (model_dir, None)
    # a card edit is fine; a weights change is not
    with open(os.path.join(model_dir, "README.md"), "a") as f:
        f.write("\nhand edit\n")
    assert pub.verify_staged(root, T, what="model") == (model_dir, None)
    with open(os.path.join(model_dir, "metrics.json"), "a") as f:
        f.write(" ")
    with pytest.raises(pub.PublishError, match="staged files changed since staging"):
        pub.verify_staged(root, T, what="model")


def test_push_staged_full_staging_accepts_any_what(tmp_path):
    root, *_ = make_root(tmp_path)
    model_dir, dataset_dir = pub.stage(root, T)
    for what in ("both", "model", "dataset"):
        assert pub.verify_staged(root, T, what=what) == (model_dir, dataset_dir)


def test_cli_what_model_stages_and_reports_model_only(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    assert pub.main(["--label_type", T, "--out", root, "--what", "model"]) == 0
    out = capsys.readouterr().out
    assert "(model only)" in out and "dataset card" not in out
    assert pub.main(["--label_type", T, "--out", root, "--push-staged"]) == 1
    assert "pass --what model" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------------
# Private deployments never reach a published file
# ---------------------------------------------------------------------------------------------------------------------

PUBLIC = {"teaneck", "chicago", "oradell"}
LISTS = {"public": PUBLIC, "non_public_terms": {"zurich", "zurich-infra3d", "Zurich", "Zürich, Switzerland",
                                                "sidewalk-zurich.cs.washington.edu", "columbia-sc", "Columbia",
                                                "switzerland"},
         "source": "tests", "read_at": "2026-10-22T00:00:00+00:00", "mode": "api"}


def _grep(folder, needle, flags=re.I):
    hits = []
    for rel, _ in pub.list_files(folder):
        if rel.lower().endswith(pub.TEXT_SUFFIXES):
            if re.search(needle, open(os.path.join(folder, rel), errors="ignore").read(), flags):
                hits.append(rel)
    return hits


def _add_private_city(root, label_type=T, city="zurich"):
    """Put a private city into the per-city metrics, the prior's source and the held-out split of a make_root() tree."""
    base = os.path.join(root, label_type)
    metrics = json.load(open(os.path.join(base, "metrics.json")))
    metrics["per_city"] = {"test_heldout": {"min_city_n": 1, "cities": {city: {"n": 2, "auc": 0.8},
                                                                        "oradell": {"n": 2, "auc": 0.9}},
                                            "below_min_n": {f"{city}-infra3d": 0}}}
    json.dump(metrics, open(os.path.join(base, "metrics.json"), "w"))
    rep_path = os.path.join(base, "calibration", "calibration_report.json")
    rep = json.load(open(rep_path))
    rep["prior_source"] = {"source": f"rawLabels of chicago, teaneck, {city}, {city}-infra3d, human votes only",
                           "cities": ["chicago", "teaneck", city, f"{city}-infra3d"]}
    json.dump(rep, open(rep_path, "w"), default=float)
    priors = json.load(open(os.path.join(root, "deployment_priors.json")))
    priors[label_type]["source"] = f"rawLabels of chicago, teaneck, {city}"
    priors[label_type]["cities"] = ["chicago", "teaneck", city]
    json.dump(priors, open(os.path.join(root, "deployment_priors.json"), "w"))
    split = pd.read_csv(os.path.join(base, "split.csv"))
    split.loc[split["split"] == "test_heldout", "city"] = city      # the private city is a held-out city
    split["label_uid"] = split["city"] + ":" + split["label_id"].astype(str)
    split.to_csv(os.path.join(base, "split.csv"), index=False)
    key = json.load(open(os.path.join(base, "dinov2", "run_key.json")))
    key["split_sha256"] = hashlib.sha256(open(os.path.join(base, "split.csv"), "rb").read()).hexdigest()
    json.dump(key, open(os.path.join(base, "dinov2", "run_key.json"), "w"))
    ev = split[split["split"].isin(["val", "test", "test_heldout"])]
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                  "p_correct": [0.7, 0.4, 0.8, 0.3, 0.9, 0.2][:len(ev)], "p_incorrect": [0.3, 0.6, 0.2, 0.7, 0.1, 0.8][:len(ev)]}
                 ).to_csv(os.path.join(base, "predictions.csv"), index=False)


def test_scrub_source_text_and_city_lists():
    assert pub.scrub_source_text("rawLabels of a, zurich, b, human votes only", {"zurich"}) == \
        "rawLabels of a, b, human votes only (and 1 non-public deployment)"
    assert pub.scrub_source_text("rawLabels of zurich, zurich-infra3d, a", {"zurich", "zurich-infra3d"}) == \
        "rawLabels of a (and 2 non-public deployments)"
    assert pub.scrub_source_text("no cities here", {"zurich"}) == "no cities here"
    assert pub.scrub_city_list(["a", "zurich", "b"], {"a", "b"}) == (["a", "b"], 1)
    out = pub.redact_metadata({"cities": ["a", "zurich"], "heldout_cities": ["zurich"], "city_source": {"a": 1, "zurich": 2},
                               "source": "rawLabels of a, zurich"}, {"a"})
    assert out == {"cities": ["a"], "n_non_public_cities": 1, "heldout_cities": [], "n_non_public_heldout": 1,
                   "city_source": {"a": 1}, "source": "rawLabels of a (and 1 non-public deployment)"}


def test_model_staging_removes_private_cities_and_pools_their_metrics(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    _add_private_city(root)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    out = capsys.readouterr().out
    assert "survivor gate: no non-public deployment name" in out
    assert not _grep(model_dir, "zurich|columbia|private-")
    metrics = json.load(open(os.path.join(model_dir, "metrics.json")))
    block = metrics["per_city"]["test_heldout"]
    assert set(block["cities"]) == {"oradell"} and block["below_min_n"] == {}
    assert block["non_public"]["n_cities"] == 2 and block["non_public"]["n"] == 2      # pooled from predictions.csv
    assert "auc" in block["non_public"]
    rep = json.load(open(os.path.join(model_dir, "calibration_report.json")))
    assert rep["prior_source"]["cities"] == ["chicago", "teaneck"] and rep["prior_source"]["n_non_public_cities"] == 2
    assert rep["prior_source"]["source"].endswith("(and 2 non-public deployments)")
    card = open(os.path.join(model_dir, "README.md")).read()
    assert "test_heldout 2 (1 non-public deployment)" in card
    record = json.load(open(os.path.join(root, T, "publish", pub.STAGING_RECORD)))
    assert record["cities"]["public"] == sorted(PUBLIC) and record["cities"]["n_non_public_terms"] == 8
    assert "zurich" not in json.dumps(record)


@pytest.mark.parametrize("source", [
    "rawLabels of chicago, teaneck, zurich (Zurich, Switzerland)",      # display case
    "rawLabels of chicago, zurich; e.g. zurich_123.webp",               # crop name, underscore
    "rawLabels from sidewalk-zurich.cs.washington.edu",                 # hostname
    "rawLabels of chicago, teaneck, columbia-sc, human votes only",     # id only in free text, no structured slot
])
def test_survivor_gate_catches_names_the_redaction_misses(tmp_path, source):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    p = os.path.join(root, T, "calibration", "calibration_report.json")
    r = json.load(open(p))
    r["prior_source"] = {"source": source}
    json.dump(r, open(p, "w"), default=float)
    try:
        model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    except pub.PublishError as e:
        assert "survive in the staged files" in str(e)
        return
    assert not _grep(model_dir, "zurich|columbia")


def test_overrides_naming_a_private_city_are_refused(tmp_path):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    json.dump({"failure_modes": ["Tram rails in Zurich read as curb ramps."]},
              open(os.path.join(root, T, "card_overrides.json"), "w"))
    with pytest.raises(pub.PublishError, match="survive in the staged files"):
        pub.stage(root, T, what="model", lists=LISTS)


def test_survivor_gate_is_independent_of_the_redaction(tmp_path, monkeypatch):
    """A no-op redaction must still be caught: the gate reads bytes, not slots."""
    root, *_ = make_root(tmp_path)
    _add_private_city(root)
    _remove_crops(root)
    monkeypatch.setattr(pub, "redact_metadata", lambda obj, public: obj)
    monkeypatch.setattr(pub, "pool_private_city_metrics", lambda m, *a: m)
    with pytest.raises(pub.PublishError, match="survive in the staged files"):
        pub.stage(root, T, what="model", lists=LISTS)
    # and the gate alone, on a folder
    d = tmp_path / "f"
    d.mkdir()
    (d / "x.json").write_text('{"a": "Seen in ZURICH once"}')
    (d / "w.safetensors").write_bytes(b"zurich")                 # binary: not a text file
    with pytest.raises(pub.PublishError, match="x.json: 'zurich'"):
        pub.check_no_private_names(str(d), {"zurich"})
    assert pub.check_no_private_names(str(d), {"nowhere"}) == 1
    assert pub.check_no_private_names(str(d), {"DC"}) == 0        # short display name: skipped, not an id


def test_dataset_staging_refuses_a_private_split_and_scrubs_the_rest(tmp_path):
    root, *_ = make_root(tmp_path)
    _add_private_city(root)
    with pytest.raises(pub.PublishError, match="not public; a dataset repo would name them"):
        pub.stage(root, T, lists=LISTS)
    # all-public split, but the prior, the generation run and the manifest still name a private deployment
    root, *_ = make_root(tmp_path / "b")
    pri = json.load(open(os.path.join(root, "deployment_priors.json")))
    pri[T]["source"] = "rawLabels of teaneck, chicago, zurich, human votes only"
    pri[T]["cities"] = ["chicago", "teaneck", "zurich"]
    json.dump(pri, open(os.path.join(root, "deployment_priors.json"), "w"))
    g = json.load(open(os.path.join(root, T, "generation_run.json")))
    g["cities"] = ["chicago", "teaneck", "zurich"]
    g["city_source"] = {"zurich": "api"}
    json.dump(g, open(os.path.join(root, T, "generation_run.json"), "w"))
    m = pd.read_csv(os.path.join(root, T, "manifest.csv"))
    extra = m.iloc[[0]].assign(city="zurich", label_uid="zurich:5", label_id=5, crop_path="correct/zurich_5.webp")
    pd.concat([m, extra]).to_csv(os.path.join(root, T, "manifest.csv"), index=False)
    model_dir, dataset_dir = pub.stage(root, T, lists=LISTS)
    assert not _grep(dataset_dir, "zurich") and not _grep(model_dir, "zurich")
    assert "zurich" not in pd.read_csv(os.path.join(dataset_dir, "manifest.csv"))["city"].values
    gen = json.load(open(os.path.join(dataset_dir, "generation_run.json")))
    assert gen["cities"] == ["chicago", "teaneck"] and gen["n_non_public_cities"] == 1 and gen["city_source"] == {}
    prior = json.load(open(os.path.join(dataset_dir, "deployment_priors.json")))[T]
    assert prior["cities"] == ["chicago", "teaneck"] and "non-public" in prior["source"]


def test_push_staged_reruns_the_gate_and_needs_the_record(tmp_path):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    assert pub.verify_staged(root, T, what="model", lists=LISTS) == (model_dir, None)
    # a fresher list that now knows a name sitting in the card refuses
    bigger = dict(LISTS, non_public_terms=LISTS["non_public_terms"] | {"dinov2"})
    with pytest.raises(pub.PublishError, match="survive in the staged files"):
        pub.verify_staged(root, T, what="model", lists=bigger)
    # a list that no longer calls a staged city public refuses
    smaller = dict(LISTS, public=PUBLIC - {"teaneck"})
    with pytest.raises(pub.PublishError, match="public city list that names 1 deployment"):
        pub.verify_staged(root, T, what="model", lists=smaller)
    # a staging from before the gate (no cities record) refuses
    rec_path = os.path.join(root, T, "publish", pub.STAGING_RECORD)
    rec = json.load(open(rec_path))
    del rec["cities"]
    json.dump(rec, open(rec_path, "w"))
    with pytest.raises(pub.PublishError, match="before the private-deployment gate"):
        pub.verify_staged(root, T, what="model", lists=LISTS)


def test_city_lists_from_files_and_refusals(tmp_path):
    f = tmp_path / "public.txt"
    f.write_text("teaneck\n# comment\nchicago  # also\n\n")
    lists = _real_city_lists(public_cities_file=str(f))
    assert lists["public"] == {"teaneck", "chicago"} and lists["non_public_terms"] == set()
    j = tmp_path / "cities.json"
    j.write_text(json.dumps({"public": ["teaneck"], "non_public_terms": ["zurich", "Zurich"]}))
    lists = _real_city_lists(public_cities_file=str(j))
    assert lists["public"] == {"teaneck"} and lists["non_public_terms"] == {"zurich", "Zurich"}
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(pub.PublishError, match="lists no public city ids"):
        _real_city_lists(public_cities_file=str(tmp_path / "empty.txt"))
    (tmp_path / "bad.json").write_text('{"nope": 1}')
    with pytest.raises(pub.PublishError, match="expected"):
        _real_city_lists(public_cities_file=str(tmp_path / "bad.json"))
    with pytest.raises(pub.PublishError, match="could not read the city list"):
        _real_city_lists(cities_api="http://127.0.0.1:9/nope", public_cities_file=None)


def test_city_lists_from_the_api_shape(monkeypatch):
    import requests

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"cities": [
                {"city_id": "seattle-wa", "visibility": "public", "city_name_short": "Seattle", "country_id": "usa",
                 "url": "https://sidewalk-sea.cs.washington.edu"},
                {"city_id": "zurich", "visibility": "private", "city_name_short": "Zurich", "country_id": "switzerland",
                 "city_name_formatted": "Zürich, Switzerland", "url": "https://sidewalk-zurich.cs.washington.edu/"},
                {"city_id": "mystery", "city_name_short": "M", "country_id": "usa"}]}      # no visibility: private

    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    lists = _real_city_lists(cities_api="http://x/cities")
    assert lists["public"] == {"seattle-wa"}
    assert lists["non_public_terms"] == {"zurich", "Zurich", "Zürich, Switzerland", "sidewalk-zurich.cs.washington.edu",
                                         "mystery", "M", "switzerland"}      # usa is a public country: not a term
    assert lists["mode"] == "api"


def test_cli_needs_a_city_list_and_what_dataset_stages_both(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    f = tmp_path / "public.txt"
    f.write_text("teaneck\nchicago\noradell\n")
    assert pub.main(["--label_type", T, "--out", root, "--what", "dataset", "--public-cities", str(f)]) == 0
    out = capsys.readouterr().out
    assert "and " in out and "(model only)" not in out
    record = json.load(open(os.path.join(root, T, "publish", pub.STAGING_RECORD)))
    assert record["what"] == "both" and record["cities"]["source"].endswith("public.txt")
    _remove_crops(root)
    assert pub.main(["--label_type", T, "--out", root, "--what", "model", "--public-cities", str(f)]) == 0
    assert "(model only)" in capsys.readouterr().out
    pub.city_lists = _real_city_lists
    try:
        assert pub.main(["--label_type", T, "--out", root, "--what", "model", "--cities-api", "http://127.0.0.1:9/x"]) == 1
    finally:
        pass
    assert "could not read the city list" in capsys.readouterr().err


def test_retry_command_works_for_a_model_only_staging(tmp_path):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    cmd = pub.retry_command(T, model_dir, pub.DEFAULT_PREFIX, [{"repo_type": "model"}], allow_todo=True)
    assert cmd == f"python publish_to_hf.py --label_type {T} --out {root} --push-staged --what model --yes --allow-todo"


def test_card_links_issue_numbers_and_flags_the_dataset_link():
    assert hf_cards.link_issues("under review (issue #56) and #7") == (
        "under review (issue [#56](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/56)) and "
        "[#7](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/7)")
    assert hf_cards.link_issues("[#56](x) already linked") == "[#56](x) already linked"
    assert hf_cards._city_list_text(["a", "b"], 2) == "a, b and 2 non-public deployments"
    assert hf_cards._city_list_text([], 1) == "1 non-public deployment"


@pytest.mark.parametrize("text", ["Tram rails in Zürich read as curb ramps.", "TRAM RAILS IN ZÜRICH", "seen in Zurich",
                                  "the Swiss deployment (Switzerland)", "ZÜRICH, SWITZERLAND"])
def test_gate_folds_accents_and_case_and_knows_countries(tmp_path, text):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    json.dump({"failure_modes": [text]}, open(os.path.join(root, T, "card_overrides.json"), "w"))
    with pytest.raises(pub.PublishError, match="survive in the staged files"):
        pub.stage(root, T, what="model", lists=LISTS)


def test_gate_reads_every_utf8_file_and_file_names(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    (d / "SHA256SUMS").write_text("abc  test/correct/zurich_5.webp\n")          # no suffix
    with pytest.raises(pub.PublishError, match="SHA256SUMS: 'zurich'"):
        pub.check_no_private_names(str(d), {"zurich"})
    (d / "SHA256SUMS").write_text("clean\n")
    (d / "test").mkdir()
    (d / "test" / "zurich_5.webp").write_bytes(b"RIFF\xff\xfe binary")        # name hit, content undecodable
    with pytest.raises(pub.PublishError, match=r"zurich_5.webp \(file name\): 'zurich'"):
        pub.check_no_private_names(str(d), {"zurich"})
    (d / "test" / "zurich_5.webp").unlink()
    (d / "weights.bin").write_bytes(b"\xff\xfe" + "zürich".encode())           # undecodable: not read
    assert pub.check_no_private_names(str(d), {"zurich"}) == 1
    assert pub.fold_text("ZÜRICH, Schweiz") == "zurich, schweiz"


def test_push_staged_refuses_plain_file_lists_and_term_drift(tmp_path, capsys):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    # H4: a hand-edited card naming a private deployment, verified with a plain id-per-line file: refused as plain
    with open(os.path.join(model_dir, "README.md"), "a") as f:
        f.write("\nValidated in Zurich too.\n")
    plain = dict(LISTS, non_public_terms=set(), mode="plain")
    with pytest.raises(pub.PublishError, match="plain id-per-line file"):
        pub.verify_staged(root, T, what="model", lists=plain)
    # with the real terms the edit is caught by the gate
    with pytest.raises(pub.PublishError, match="survive in the staged files"):
        pub.verify_staged(root, T, what="model", lists=LISTS)
    # term drift: fewer terms than recorded, or a different set, is refused unless accepted
    root, *_ = make_root(tmp_path / "b")
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    fewer = dict(LISTS, non_public_terms=LISTS["non_public_terms"] - {"Columbia"})
    with pytest.raises(pub.PublishError, match="non-public term list changed since staging"):
        pub.verify_staged(root, T, what="model", lists=fewer)
    assert pub.verify_staged(root, T, what="model", lists=fewer, accept_city_list_change=True) == (model_dir, None)
    assert "accepted" in capsys.readouterr().out
    assert pub.verify_staged(root, T, what="model", lists=LISTS) == (model_dir, None)
    f = tmp_path / "public.txt"
    f.write_text("teaneck\nchicago\noradell\n")
    assert pub.main(["--label_type", T, "--out", root, "--push-staged", "--what", "model",
                     "--public-cities", str(f)]) == 1
    assert "plain id-per-line file" in capsys.readouterr().err


def test_pooled_row_is_withheld_for_a_single_private_deployment(tmp_path):
    root, *_ = make_root(tmp_path)
    _add_private_city(root)
    _remove_crops(root)
    base = os.path.join(root, T)
    metrics = json.load(open(os.path.join(base, "metrics.json")))
    metrics["per_city"]["test_heldout"]["below_min_n"] = {}          # leaves zurich alone in the pool
    json.dump(metrics, open(os.path.join(base, "metrics.json"), "w"))
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    block = json.load(open(os.path.join(model_dir, "metrics.json")))["per_city"]["test_heldout"]
    assert block["non_public"] == {"n_cities": 1, "note": "metrics withheld: a single non-public deployment"}


def test_whole_card_links_issue_numbers(tmp_path):
    root, *_ = make_root(tmp_path)
    _remove_crops(root)
    model_dir, _ = pub.stage(root, T, what="model", lists=LISTS)
    card = open(os.path.join(model_dir, "README.md")).read()
    assert "(issues [#2](https://github.com/ProjectSidewalk/sidewalk-validator-ai/issues/2) and [#7](" in card
    assert re.search(r"(?<![\w\[/])#\d+\b", card) is None                   # no bare issue number of this repo
    assert "policy decision (issues" in card and ").  Labels" not in card
