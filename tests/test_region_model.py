"""region_model.py: the vocabulary helpers (always run) and the region-conditioned DINOv2 (torch + transformers;
skipped on the CPU-only CI)."""
import doctest
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import region_model as rm
import regions

ID2LABEL = {0: "correct", 1: "incorrect"}
LABEL2ID = {"correct": 0, "incorrect": 1}
TABLE = regions.country_table([], {})


def test_vocab_sorted_with_unk_first():
    df = pd.DataFrame({"city": ["spgg", "seattle-wa", "amsterdam", "cdmx"]})
    assert rm.region_vocab_from_split(df, "country", TABLE) == ["<unk>", "mexico", "netherlands", "usa"]
    assert rm.region_vocab_from_split(df, "region", TABLE) == ["<unk>", "europe", "latin-america", "north-america"]
    with pytest.raises(ValueError):
        rm.region_value("cdmx", "continent", TABLE)


def test_unknown_region_maps_to_zero():
    assert rm.region_ids(["usa", "atlantis", "mexico"], ["<unk>", "mexico", "usa"]) == [2, 0, 1]


def test_city_region_ids_val_outside_vocab_is_unk():
    vocab = rm.region_vocab_from_split(pd.DataFrame({"city": ["seattle-wa"]}), "country", TABLE)
    ids = rm.city_region_ids(["seattle-wa", "amsterdam", "nowhere"], vocab, "country", TABLE)
    assert ids == {"amsterdam": 0, "nowhere": 0, "seattle-wa": 1}
    assert len(rm.vocab_sha256(vocab)) == 64 and rm.vocab_sha256(vocab) != rm.vocab_sha256(vocab + ["x"])


def test_build_for_training_refuses_frames_without_city():
    args = SimpleNamespace(region_conditioning="country", cities_json="unused.json", region_dim=4, region_dropout=0.1)
    with pytest.raises(SystemExit, match="city column"):
        rm.build_for_training(args, "base", pd.DataFrame({"image": ["a"]}), None, ID2LABEL, LABEL2ID, None)


def test_doctests():
    assert doctest.testmod(rm).failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# torch + transformers
# ---------------------------------------------------------------------------------------------------------------------

def tiny(vocab=("<unk>", "a", "b"), dropout=0.15):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    cfg_cls, model_cls = rm.build_classes()
    cfg = cfg_cls(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16,
                  region_vocab=list(vocab), region_dim=4, region_dropout=dropout, id2label=ID2LABEL, label2id=LABEL2ID)
    torch.manual_seed(0)
    return model_cls(cfg)


def test_forward_shapes_and_loss():
    torch = pytest.importorskip("torch")
    m = tiny().eval()
    out = m(pixel_values=torch.randn(3, 3, 32, 32), region_ids=torch.tensor([0, 1, 2]), labels=torch.tensor([0, 1, 0]))
    assert out.logits.shape == (3, 2) and out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert m.classifier.in_features == 2 * 32 + 4


def test_region_ids_none_equals_unk():
    torch = pytest.importorskip("torch")
    m = tiny().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = m(pixel_values=x).logits
        b = m(pixel_values=x, region_ids=torch.zeros(2, dtype=torch.long)).logits
    assert torch.allclose(a, b)


def test_dropout_replaces_about_p_in_train_and_none_in_eval():
    torch = pytest.importorskip("torch")
    m = tiny(dropout=0.15)
    ids = torch.ones(2000, dtype=torch.long)
    torch.manual_seed(1)
    m.train()
    share = float((m.region_input(ids, 2000, "cpu") == 0).float().mean())
    assert abs(share - 0.15) < 0.03
    m.eval()
    assert int((m.region_input(ids, 2000, "cpu") == 0).sum()) == 0


def test_embedding_changes_logits_between_regions():
    torch = pytest.importorskip("torch")
    m = tiny().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a = m(pixel_values=x, region_ids=torch.tensor([1])).logits
        b = m(pixel_values=x, region_ids=torch.tensor([2])).logits
    assert not torch.allclose(a, b)


def test_pooling_matches_dinov2_for_image_classification():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    m = tiny().eval()
    plain = transformers.Dinov2ForImageClassification(transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16, num_labels=2)).eval()
    plain.dinov2.load_state_dict(m.dinov2.state_dict())
    with torch.no_grad():   # zero the region part of the head, copy the image part: the logits must agree
        m.classifier.weight[:, 64:] = 0
        plain.classifier.weight.copy_(m.classifier.weight[:, :64])
        plain.classifier.bias.copy_(m.classifier.bias)
        x = torch.randn(2, 3, 32, 32)
        assert torch.allclose(m(pixel_values=x).logits, plain(pixel_values=x).logits, atol=1e-5)


def test_save_load_roundtrip_keeps_vocab(tmp_path):
    torch = pytest.importorskip("torch")
    m = tiny().eval()
    m.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["region_vocab"] == ["<unk>", "a", "b"]
    _, model_cls = rm.build_classes()
    m2 = model_cls.from_pretrained(tmp_path).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(m(pixel_values=x, region_ids=torch.tensor([2])).logits,
                              m2(pixel_values=x, region_ids=torch.tensor([2])).logits)


def test_from_base_copies_backbone_weights(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    base = transformers.Dinov2ForImageClassification(transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16, num_labels=2))
    base.save_pretrained(tmp_path)
    _, model_cls = rm.build_classes()
    m = model_cls.from_base(str(tmp_path), ["<unk>", "mexico"], 4, 0.2, "country", ID2LABEL, LABEL2ID)
    for k, v in base.dinov2.state_dict().items():
        assert torch.equal(m.dinov2.state_dict()[k], v), k
    assert m.config.region_vocab == ["<unk>", "mexico"] and m.config.region_dropout == pytest.approx(0.2)
    assert m.config.id2label == ID2LABEL


def test_wrap_collate_adds_region_ids():
    torch = pytest.importorskip("torch")

    def parent(examples):
        return {"labels": torch.tensor([e["label"] for e in examples])}
    collate = rm.wrap_collate(parent, {"cdmx": 1, "amsterdam": 2})
    batch = collate([{"label": 0, "city": "cdmx"}, {"label": 1, "city": "amsterdam"}, {"label": 1, "city": "x"}])
    assert batch["region_ids"].tolist() == [1, 2, 0] and batch["labels"].tolist() == [0, 1, 1]


def test_run_inference_on_two_fake_crops_writes_prediction_columns(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image

    import evaluate
    m = tiny(vocab=("<unk>", "mexico"))
    ckpt = tmp_path / "ckpt"
    m.save_pretrained(ckpt)
    transformers.BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(ckpt)
    rows = []
    for i, (city, split) in enumerate([("cdmx", "test"), ("amsterdam", "test"), ("cdmx", "train")]):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((40, 40, 3), 30 * i, dtype=np.uint8)).save(p)
        rows.append({"label_uid": f"{city}:{i}", "city": city, "split": split, "label_class": "correct",
                     "crop_path": str(p)})
    manifest = pd.DataFrame(rows)
    out = rm.run_inference(manifest, str(ckpt), ("test",), TABLE)
    assert list(out.columns) == evaluate.PREDICTION_COLUMNS and len(out) == 2
    assert np.allclose(out.p_correct + out.p_incorrect, 1)
    unk = rm.run_inference(manifest, str(ckpt), ("test",), TABLE, force_unk=True)
    # amsterdam is outside the vocabulary, so it is unknown either way; cdmx changes when forced to unknown
    assert unk.p_incorrect.iloc[1] == pytest.approx(out.p_incorrect.iloc[1])
    assert unk.p_incorrect.iloc[0] != pytest.approx(out.p_incorrect.iloc[0])
