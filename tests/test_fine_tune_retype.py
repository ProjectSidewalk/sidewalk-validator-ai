"""fine_tune_retype.py: the run key and resume guard, the frames, the validation metrics, the recipe constants, and
(torch only) the loss, a tiny random DINOv2 with the 8-class head, and one short Trainer run end to end."""
import doctest
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import fine_tune_classifier as ftc
import fine_tune_retype as fr
import retype as rt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def test_doctests():
    failed, _ = doctest.testmod(fr)
    assert failed == 0


def args_for(split, **kw):
    a = dict(split=str(split), seed=42, epochs=12, best_metric="auc_claimed", negatives="complement",
             class_weight_power=0.5, cap_per_class=None)
    a.update(kw)
    return SimpleNamespace(**a)


def test_run_key_covers_negatives_weights_cap_and_the_split(tmp_path):
    split = tmp_path / "retype_split.csv"
    split.write_text("row_uid,split\na:1@CurbRamp,train\n")
    base = fr.retype_run_key(args_for(split))
    assert base["split_sha256"] == ftc.file_sha256(str(split)) and base["classes"] == list(rt.CLASSES)
    for change in ({"negatives": "other"}, {"class_weight_power": 0.0}, {"cap_per_class": 5000}, {"seed": 1},
                   {"epochs": 25}, {"best_metric": "loss"}):
        assert fr.retype_run_key(args_for(split, **change)) != base, change
    split.write_text("row_uid,split\na:2@CurbRamp,train\n")
    assert fr.retype_run_key(args_for(split))["split_sha256"] != base["split_sha256"]


def test_resume_refuses_the_other_legs_checkpoints(tmp_path):
    split = tmp_path / "retype_split.csv"
    split.write_text("row_uid,split\na:1@CurbRamp,train\n")
    out = tmp_path / "retype" / "complement"
    assert ftc.resume_checkpoint(str(out), fr.retype_run_key(args_for(split))) is None
    ck = out / "checkpoint-100"
    ck.mkdir()
    (ck / "trainer_state.json").write_text("{}")
    assert ftc.resume_checkpoint(str(out), fr.retype_run_key(args_for(split))) == str(ck)
    with pytest.raises(SystemExit, match="negatives"):
        ftc.resume_checkpoint(str(out), fr.retype_run_key(args_for(split, negatives="other")))


def write_split(path, rows):
    pd.DataFrame(rows, columns=["row_uid", "split", "claimed_type", "target_type", "is_correct",
                                "crop_path"]).to_csv(path, index=False)


def test_load_frames_targets_and_cap(tmp_path):
    p = tmp_path / "s.csv"
    rows = [(f"c:{i}@CurbRamp", "train", "CurbRamp", "CurbRamp", 1, f"CurbRamp/{i}.webp") for i in range(6)]
    rows += [("c:9@Obstacle", "train", "Obstacle", "", 0, "Obstacle/9.webp"),
             ("c:8@Other", "val", "Other", "Other", 1, "Occlusion/8.webp"),
             ("c:7@CurbRamp", "test", "CurbRamp", "CurbRamp", 1, "x.webp"),
             ("c:6@NoCurbRamp", "test_retype", "NoCurbRamp", "CurbRamp", 0, "/abs/6.webp")]
    write_split(p, rows)
    tr, va, dropped = fr.load_frames(str(p), "/root", cap_per_class=4)
    assert dropped == {"CurbRamp": 2} and len(tr) == 5 and len(va) == 1
    assert (tr.label == fr.IGNORE).sum() == 1 and tr[tr.label == fr.IGNORE].claimed.tolist() == [2]
    assert va.label.tolist() == [7] and va.image.tolist() == ["/root/Occlusion/8.webp"]
    assert fr.train_counts(tr) == {**{c: 0 for c in rt.CLASSES}, "CurbRamp": 4}
    write_split(p, rows + [rows[0]])
    with pytest.raises(ValueError, match="row_uid"):
        fr.load_frames(str(p), "/root")


def test_compute_retype_metrics_known_values():
    probs = np.full((6, 8), 0.01)
    # rows: claimed CurbRamp x4 (2 correct, 2 incorrect), claimed Obstacle x2 (both correct)
    probs[0, 0], probs[1, 0], probs[2, 0], probs[3, 0] = 0.9, 0.6, 0.3, 0.7
    probs[4, 2], probs[5, 1] = 0.9, 0.9
    logits = np.log(probs)
    labels = [0, 0, -100, -100, 2, 2]
    claimed = [0, 0, 0, 0, 2, 2]
    is_correct = [1, 1, 0, 0, 1, 1]
    m = fr.compute_retype_metrics(logits, labels, claimed, is_correct)
    assert m["top1_acc"] == pytest.approx(3 / 4)          # row 5 says NoCurbRamp; -100 rows excluded
    assert m["auc_claimed_CurbRamp"] == pytest.approx(0.75)  # 0.9, 0.6 vs 0.3, 0.7
    assert "auc_claimed_Obstacle" not in m and m["auc_claimed"] == pytest.approx(0.75)
    assert 0 < m["macro_f1"] < 1
    assert fr.compute_retype_metrics(logits[:2], [0, 0], [0, 0], [1, 1])["auc_claimed"] == 0.0


def test_auc_claimed_is_row_weighted_over_the_validator_types_only():
    """CurbRamp (8 val rows, AUC 1.0) and NoCurbRamp (2 rows, AUC 0.0) average to 0.8, not 0.5; NoSidewalk (an
    extra, AUC 0.0) is reported but does not select."""
    rows = []  # (claimed index, is_correct, p(claimed))
    rows += [(0, 1, 0.9)] * 4 + [(0, 0, 0.2)] * 4
    rows += [(1, 1, 0.2), (1, 0, 0.9)]
    rows += [(5, 1, 0.2), (5, 0, 0.9)]
    probs = np.full((len(rows), 8), 0.01)
    for k, (c, _, pc) in enumerate(rows):
        probs[k, c] = pc
    claimed = [r[0] for r in rows]
    is_correct = [r[1] for r in rows]
    labels = [c if ok else -100 for c, ok, _ in rows]
    m = fr.compute_retype_metrics(np.log(probs), labels, claimed, is_correct)
    assert m["auc_claimed_CurbRamp"] == 1.0 and m["auc_claimed_NoCurbRamp"] == 0.0
    assert m["auc_claimed_NoSidewalk"] == 0.0 and "NoSidewalk" not in rt.VALIDATOR_TYPES
    assert m["auc_claimed"] == pytest.approx(0.8)
    only_extra = fr.compute_retype_metrics(np.log(probs[-2:]), labels[-2:], claimed[-2:], is_correct[-2:])
    assert only_extra["auc_claimed"] == 0.0 and "auc_claimed_NoSidewalk" in only_extra


def test_trainer_metrics_unpacks_the_label_tuple():
    logits = np.log(np.full((2, 8), 1 / 8))
    pred = SimpleNamespace(predictions=logits, label_ids=(np.array([0, -100]), np.array([0, 0]), np.array([1, 0])))
    assert fr.trainer_metrics(pred)["auc_claimed"] == pytest.approx(0.5)
    assert fr.LABEL_NAMES == ["labels", "claimed", "is_correct"]


def test_recipe_constants_match_round_3():
    src = open(os.path.join(REPO, "fine_tune_classifier.py")).read()
    assert "scale=(0.7, 1.0)" in src and fr.CROP_SCALE == (0.7, 1.0)
    assert re.search(r"per_device_train_batch_size=8\b", src) and fr.TRAIN_BATCH_SIZE == 8
    assert re.search(r"gradient_accumulation_steps=4\b", src) and fr.GRAD_ACCUM == 4
    assert re.search(r"learning_rate=1e-5\b", src) and fr.LEARNING_RATE == 1e-5
    assert re.search(r"warmup_ratio=0.1\b", src) and fr.WARMUP_RATIO == 0.1
    own = open(os.path.join(REPO, "fine_tune_retype.py")).read()
    assert "use_fast=False" in own and "remove_unused_columns=False" in own and "label_names=LABEL_NAMES" in own
    assert fr.build_parser().parse_args(["--split", "s", "--crop-root", "r", "--output-dir", "o"]).epochs == 12


# ---------------------------------------------------------------------------------------------------------------------
# torch only (skipped in CI, which installs no torch)
# ---------------------------------------------------------------------------------------------------------------------

def test_retype_loss_complement_other_and_scaling():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    labels = torch.tensor([3, -100, 0, -100])
    claimed = torch.tensor([3, 1, 0, 7])
    is_correct = torch.tensor([1, 0, 1, 0])
    p = torch.softmax(logits, -1)
    ce = -torch.log(p[torch.arange(4), torch.tensor([3, 0, 0, 0])])
    comp = -torch.log(1 - p[torch.arange(4), claimed])
    want = (ce[0] + comp[1] + ce[2] + comp[3]) / 4
    assert torch.isclose(fr.retype_loss(logits, labels, claimed, is_correct), want)
    other = -torch.log(p[:, rt.LABEL2ID["Other"]])
    want_o = (ce[0] + other[1] + ce[2] + other[3]) / 4
    assert torch.isclose(fr.retype_loss(logits, labels, claimed, is_correct, negatives="other"), want_o)
    w = torch.ones(8)
    w[3] = 2.0
    want_w = (2 * ce[0] + comp[1] + ce[2] + comp[3]) / 4
    assert torch.isclose(fr.retype_loss(logits, labels, claimed, is_correct, w), want_w)
    assert torch.isclose(fr.retype_loss(logits, labels, claimed, is_correct, num_items_in_batch=8), want / 2)
    with pytest.raises(ValueError):
        fr.retype_loss(logits, labels, claimed, is_correct, negatives="nope")
    # a confident claimed probability of 1 stays finite, and equals the logit gap (log(1 + 7 e^-100) ~ 100)
    big = torch.full((1, 8), -50.0)
    big[0, 2] = 50.0
    big_loss = fr.retype_loss(big, torch.tensor([-100]), torch.tensor([2]), torch.tensor([0]))
    assert torch.isfinite(big_loss) and big_loss.item() == pytest.approx(100.0 - np.log(7.0), rel=1e-4)


def complement_at_gap(torch, gap):
    """Loss and logit gradient of one incorrect row whose claimed logit leads the other seven (all 0) by ``gap``."""
    logits = torch.zeros(1, 8)
    logits[0, 2] = gap
    logits.requires_grad_()
    loss = fr.retype_loss(logits, torch.tensor([-100]), torch.tensor([2]), torch.tensor([0]))
    loss.backward()
    return loss.detach(), logits.grad


@pytest.mark.parametrize("gap", [5, 16, 20, 30])
def test_complement_loss_keeps_its_gradient_on_confident_wrong_rows(gap):
    """The exact form -log sum_{j != claimed} p_j = gap - log 7 (+ log(1 + 7 e^-gap)) has no saturation: the
    gradient on the claimed logit tends to 1 and never collapses to 0 (the old clamp made it exactly 0
    from a gap of 16 on)."""
    torch = pytest.importorskip("torch")
    loss, grad = complement_at_gap(torch, gap)
    want = gap - np.log(7.0) + np.log1p(7.0 * np.exp(-gap))
    assert loss.item() == pytest.approx(want, rel=1e-5)
    assert grad.norm().item() > 0.5
    assert grad[0, 2].item() > 0.95  # d loss / d claimed logit tends to 1 as the gap grows


def test_complement_loss_keeps_growing_with_the_gap():
    torch = pytest.importorskip("torch")
    assert complement_at_gap(torch, 30)[0] > complement_at_gap(torch, 16)[0]
    assert complement_at_gap(torch, 20)[1].abs().sum() > 0 and complement_at_gap(torch, 30)[1].abs().sum() > 0


def test_other_leg_weights_negatives_one_whatever_others_positive_count():
    """--negatives other: an incorrect row costs CE(Other) with weight 1.0, so the ablation's negatives do not move
    when Other gains positives (and with them a class weight well above 1)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    logits = torch.randn(3, 8)
    labels, claimed, is_correct = torch.tensor([-100] * 3), torch.tensor([0, 1, 4]), torch.tensor([0, 0, 0])
    counts = {**{c: 0 for c in rt.CLASSES}, "CurbRamp": 1000}
    weights = {n: rt.class_weights({**counts, "Other": n}, 0.5) for n in (0, 10)}
    assert weights[0]["Other"] == 1.0 and weights[10]["Other"] > 1.5  # the weight that must not leak in
    o = rt.LABEL2ID["Other"]
    want = -torch.log(torch.softmax(logits, -1)[:, o]).mean()
    for w in weights.values():
        w_tensor = torch.tensor([w[c] for c in rt.CLASSES])
        got = fr.retype_loss(logits, labels, claimed, is_correct, w_tensor, negatives="other")
        assert torch.isclose(got, want)


def tiny_model():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32,
                                       patch_size=16, num_labels=8, id2label=rt.ID2LABEL, label2id=rt.LABEL2ID)
    return transformers, transformers.Dinov2ForImageClassification(config)


def test_tiny_model_has_the_eight_class_head(tmp_path):
    torch = pytest.importorskip("torch")
    transformers, model = tiny_model()
    out = model(pixel_values=torch.randn(2, 3, 32, 32)).logits
    assert out.shape == (2, 8)
    model.save_pretrained(tmp_path / "m")
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert [cfg["id2label"][str(i)] for i in range(8)] == list(rt.CLASSES)
    again = transformers.AutoModelForImageClassification.from_pretrained(tmp_path / "m")
    assert tuple(again.config.id2label[i] for i in range(8)) == rt.CLASSES


def test_main_trains_one_epoch_on_a_tiny_model(tmp_path, monkeypatch):
    """The Trainer wiring (label_names, compute_loss, metrics, best checkpoint, training_run.json) on a CPU, with the
    DINOv2-Large download replaced by a tiny random model and a small image processor."""
    pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("torchvision")
    pytest.importorskip("accelerate")
    transformers, _ = tiny_model()
    from PIL import Image
    rows = []
    for i in range(16):
        t = rt.CLASSES[i % 3]
        correct = i % 4 != 0
        path = tmp_path / "crops" / f"{i}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (40, 40), (i * 10, 50, 100)).save(path)
        rows.append((f"c:{i}@{t}", "val" if i >= 12 else "train", t, t if correct else "", int(correct),
                     f"crops/{i}.png"))
    split = tmp_path / "retype_split.csv"
    write_split(split, rows)
    processor = transformers.BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.Dinov2ForImageClassification, "from_pretrained",
                        lambda *a, **k: tiny_model()[1])
    out = tmp_path / "run"
    fr.main(["--split", str(split), "--crop-root", str(tmp_path), "--output-dir", str(out), "--epochs", "1",
             "--num-workers", "0", "--allow-cpu"])
    run = json.loads((out / "training_run.json").read_text())
    assert run["label_type"] == "retype" and run["negatives"] == "complement"
    assert run["best_model_checkpoint"].startswith("checkpoint-") and run["n_train_negatives"] == 3
    assert run["split_sha256"] == ftc.file_sha256(str(split))
    state = json.loads((out / run["best_model_checkpoint"] / "trainer_state.json").read_text())
    evals = [h for h in state["log_history"] if "eval_auc_claimed" in h]
    assert evals and "eval_top1_acc" in evals[0] and np.isfinite(evals[0]["eval_loss"])
