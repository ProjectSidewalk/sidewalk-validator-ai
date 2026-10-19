"""--target in fine_tune_classifier.py and the gold slice in evaluate.py (issue #12)."""
import doctest
from types import SimpleNamespace

import pandas as pd
import pytest

import fine_tune_classifier as ft
from evaluate import gold_labels, report
from fine_tune_classifier import load_split_frames, run_key, select_trainer, soft_target_trainer

ARGS = SimpleNamespace(seed=42, epochs=25, best_metric="macro_f1")


def _split(tmp_path, with_votes=True):
    base = tmp_path / "CurbRamp"
    base.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"label_uid": ["c:1", "c:2", "c:3"], "crop_path": ["a.webp", "b.webp", "d.webp"],
                       "label_class": ["correct", "incorrect", "correct"], "split": ["train", "train", "val"]})
    if with_votes:
        df["human_agree"] = [3, 0, 5]
        df["human_disagree"] = [0, 2, 0]
    df.to_csv(base / "split.csv", index=False)
    return df


def test_doctests():
    result = doctest.testmod(ft)
    assert result.attempted > 0 and result.failed == 0


def test_hard_default_is_unchanged(tmp_path):
    _split(tmp_path)
    train, val = load_split_frames("CurbRamp", str(tmp_path))
    assert list(train.columns) == ["image", "label"] and list(val.columns) == ["image", "label"]
    assert train.label.tolist() == [0, 1]   # ID2LABEL: 0 is "correct"
    # The default run key has exactly the keys it had before --target existed, so a resume still matches.
    key = run_key("CurbRamp", str(tmp_path), ARGS)
    assert set(key) == {"label_type", "split_sha256", "seed", "epochs", "best_metric", "base_checkpoint"}
    assert run_key("CurbRamp", str(tmp_path), SimpleNamespace(**vars(ARGS), target="hard")) == key
    assert ft.target_settings(SimpleNamespace(target="hard")) == {}


def test_hard_default_keeps_the_plain_trainer():
    class Base:
        pass
    assert select_trainer("hard", Base) is Base
    assert issubclass(select_trainer("soft", Base), Base) and select_trainer("soft", Base) is not Base


def test_soft_targets_on_train_rows_only(tmp_path):
    _split(tmp_path)
    train, val = load_split_frames("CurbRamp", str(tmp_path), "soft")
    assert train.soft_target.tolist() == [0.8, 0.25]
    assert "soft_target" not in val.columns   # validation stays hard


def test_soft_without_vote_columns_fails_loudly(tmp_path):
    _split(tmp_path, with_votes=False)
    with pytest.raises(ValueError, match="human_agree"):
        load_split_frames("CurbRamp", str(tmp_path), "soft")


def test_weighted_joins_on_label_uid_and_fails_on_missing_rows(tmp_path):
    _split(tmp_path)
    w = pd.DataFrame({"label_uid": ["c:2", "c:1", "other:1"], "w_agree": [0.0, 2.0, 9.0],
                      "w_disagree": [1.5, 0.0, 0.0]})
    train, _ = load_split_frames("CurbRamp", str(tmp_path), "weighted", w)
    assert train.soft_target.tolist() == pytest.approx([0.75, 1 / 3.5])
    with pytest.raises(ValueError, match="1 of 2 rows"):
        load_split_frames("CurbRamp", str(tmp_path), "weighted", w[w.label_uid != "c:2"])


def test_target_is_in_the_run_key(tmp_path):
    _split(tmp_path)
    vw = tmp_path / "vote_weights.csv"
    vw.write_text("label_uid,w_agree,w_disagree\nc:1,1,0\n")
    hard = run_key("CurbRamp", str(tmp_path), ARGS)
    soft = run_key("CurbRamp", str(tmp_path), SimpleNamespace(**vars(ARGS), target="soft"))
    weighted = run_key("CurbRamp", str(tmp_path),
                       SimpleNamespace(**vars(ARGS), target="weighted", vote_weights=str(vw)))
    assert soft["target"] == "soft" and soft != hard
    assert weighted["target"] == "weighted" and len(weighted["vote_weights_sha256"]) == 64
    vw.write_text("label_uid,w_agree,w_disagree\nc:1,2,0\n")
    assert run_key("CurbRamp", str(tmp_path),
                   SimpleNamespace(**vars(ARGS), target="weighted", vote_weights=str(vw))) != weighted


def test_cli_requires_vote_weights_with_weighted_only():
    with pytest.raises(SystemExit):
        ft.main(["--label_type", "CurbRamp", "--target", "weighted"])
    with pytest.raises(SystemExit):
        ft.main(["--label_type", "CurbRamp", "--vote-weights", "x.csv"])


def test_soft_target_loss_on_a_tiny_batch():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    p = torch.tensor([0.8, 0.25, 1.0])
    logp = torch.log_softmax(logits, dim=-1)
    expected = -(p * logp[:, 0] + (1 - p) * logp[:, 1])   # index 0 is "correct"
    assert torch.allclose(ft.soft_target_loss(logits, p), expected.mean())
    assert torch.allclose(ft.soft_target_loss(logits, p, num_items_in_batch=6), expected.sum() / 6)
    # 0/1 targets reduce to the hard cross-entropy on label ids (0 = correct, 1 = incorrect).
    hard = torch.tensor([1.0, 0.0, 1.0])
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    assert torch.allclose(ft.soft_target_loss(logits, hard), ce)


def test_soft_trainer_compute_loss_uses_soft_targets_and_falls_back_for_val():
    torch = pytest.importorskip("torch")

    class Base:
        model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return "hard"

    seen = {}

    def model(**inputs):
        seen.update(inputs)
        return SimpleNamespace(logits=torch.tensor([[0.0, 0.0]]))

    trainer = soft_target_trainer(Base)()
    batch = {"pixel_values": torch.zeros(1, 3), "labels": torch.tensor([0]), "soft_targets": torch.tensor([0.9])}
    loss = trainer.compute_loss(model, batch, num_items_in_batch=4)
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))   # uniform logits: log 2 whatever the target
    assert set(seen) == {"pixel_values"} and "soft_targets" in batch   # caller's batch untouched
    assert trainer.compute_loss(model, {"pixel_values": torch.zeros(1, 3), "labels": torch.tensor([0])}) == "hard"


def test_gold_slice_only_when_column_present():
    preds = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "split": ["test", "test", "test", "val"],
                          "true": ["correct", "incorrect", "correct", "correct"],
                          "p_correct": [0.9, 0.2, 0.4, 0.9], "p_incorrect": [0.1, 0.8, 0.6, 0.1]})
    assert "gold" not in report(preds)
    w = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3", "a:4"], "n_votes": [3, 2, 3, 3],
                      "min_reliability": [0.96, 0.95, 0.5, 0.99]})
    preds["gold"] = preds.label_uid.map(gold_labels(w, 0.95))
    out = report(preds)
    assert out["gold"]["n"] == 2 and out["gold"]["n_by_class"] == {"correct": 1, "incorrect": 1}


def test_evaluate_cli_gold_slice_and_unchanged_default(tmp_path):
    import json

    import evaluate
    base = tmp_path / "CurbRamp"
    base.mkdir()
    uids = ["c:1", "c:2", "c:3", "c:4"]
    pd.DataFrame({"label_uid": uids, "city": "c", "label_id": [1, 2, 3, 4], "pano_id": "p", "label_type": "CurbRamp",
                  "label_class": ["correct", "incorrect", "correct", "incorrect"], "crop_path": "x.webp",
                  "split": "test", "hard": 0}).to_csv(base / "split.csv", index=False)
    pd.DataFrame({"label_uid": uids, "split": "test", "true": ["correct", "incorrect", "correct", "incorrect"],
                  "p_correct": [0.9, 0.2, 0.6, 0.3], "p_incorrect": [0.1, 0.8, 0.4, 0.7]}).to_csv(
        base / "predictions.csv", index=False)
    evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference"])
    assert "gold" not in json.loads((base / "metrics.json").read_text())

    vw = tmp_path / "vote_weights.csv"
    pd.DataFrame({"label_uid": uids, "w_agree": 1.0, "w_disagree": 0.0, "n_votes": 3, "n_gold": [3, 3, 0, 3],
                  "min_reliability": [0.99, 0.97, 0.5, 0.9]}).to_csv(vw, index=False)
    evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference", "--vote-weights",
                   str(vw), "--gold-min-reliability", "0.96"])
    assert json.loads((base / "metrics.json").read_text())["gold"]["n"] == 2
    pd.read_csv(vw).iloc[:3].to_csv(vw, index=False)
    with pytest.raises(SystemExit, match="1 predictions have no row"):
        evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--skip-inference",
                       "--vote-weights", str(vw)])
    with pytest.raises(SystemExit):
        evaluate.main(["--label_type", "CurbRamp", "--data-dir", str(tmp_path), "--gold-min-reliability", "0.9"])
