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
