"""fine_tune_classifier.py --input-res (docs/resolution-2026-11.md): the default 224 path is unchanged (run key,
processor call, train transform, recorded preprocessing), a non-default size reaches train, val and the saved processor
alike, the parser refuses the combinations the study does not run, and (torch only) HF's DINOv2 interpolates its
position embeddings at 448 and 518 on a tiny random-weight config.

The torch tests skip in CI (no torch there); run them locally or in the GPU smoke job, as test_backbones.py's are."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

import backbones as bb
import fine_tune_classifier as ftc
from fine_tune_classifier import resume_checkpoint, run_key

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "backbones")
ARGS = SimpleNamespace(seed=42, epochs=25, best_metric="macro_f1")


def _key(tmp_path, **overrides):
    split_dir = tmp_path / "data" / "CurbRamp"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "split.csv").write_text("label_uid,split\nseattle-wa:1,train\n")
    return run_key("CurbRamp", str(tmp_path / "data"), SimpleNamespace(**{**vars(ARGS), **overrides}))


# ---------------------------------------------------------------------------------------------------------------------
# the default path is unchanged
# ---------------------------------------------------------------------------------------------------------------------

def test_default_run_key_is_unchanged_by_input_res(tmp_path):
    old = _key(tmp_path)
    assert _key(tmp_path, input_res=224) == old
    assert set(old) == {"label_type", "split_sha256", "seed", "epochs", "best_metric", "base_checkpoint"}


def test_a_non_default_input_res_adds_exactly_its_key_and_blocks_a_resume(tmp_path):
    base = _key(tmp_path, input_res=224)
    r448 = _key(tmp_path, input_res=448)
    assert set(r448) - set(base) == {"input_res"} and r448["input_res"] == 448
    out = tmp_path / "r448"
    assert resume_checkpoint(str(out), r448) is None
    (out / "checkpoint-330").mkdir()
    (out / "checkpoint-330" / "trainer_state.json").write_text("{}")
    assert resume_checkpoint(str(out), r448).endswith("checkpoint-330")   # same size: resumes
    with pytest.raises(SystemExit, match="input_res"):
        resume_checkpoint(str(out), _key(tmp_path, input_res=518))


def test_the_default_processor_is_the_call_the_script_always_made(monkeypatch):
    seen = []
    monkeypatch.setattr(bb, "build_image_processor", lambda *a, **k: seen.append((a, k)) or "processor")
    assert ftc.make_image_processor(SimpleNamespace(backbone="dinov2-large", input_res=224)) == "processor"
    assert ftc.make_image_processor(SimpleNamespace(backbone="dinov2-large")) == "processor"   # no attribute
    assert seen == [(("dinov2-large",), {}), (("dinov2-large",), {})]
    seen.clear()
    ftc.make_image_processor(SimpleNamespace(backbone="dinov2-large", input_res=448))
    assert seen == [(("dinov2-large",), {"input_res": 448})]


def test_input_sizes_keep_the_field_of_view():
    assert {r: bb.input_sizes("dinov2-large", r) for r in ftc.INPUT_RES_CHOICES} == {
        224: (256, 224), 448: (512, 448), 518: (592, 518)}
    for r in ftc.INPUT_RES_CHOICES:
        resize, crop = bb.input_sizes("dinov2-large", r)
        assert crop / resize == pytest.approx(224 / 256) and crop % 14 == 0
    assert ftc.DEFAULT_INPUT_RES == bb.get(bb.DEFAULT).crop_px
    assert bb.input_sizes("dinov2-large", None) == bb.input_sizes("dinov2-large", 224)


def test_preprocessing_at_the_default_size_is_byte_identical_and_448_differs_only_in_size():
    assert bb.preprocess_spec("dinov2-large", 224) == bb.preprocess_spec("dinov2-large")
    assert bb.preprocess_sha256("dinov2-large", 224) == bb.preprocess_sha256("dinov2-large")
    hub = json.load(open(os.path.join(FIX, "dinov2_large_preprocessor_config.json")))
    assert bb.preprocess_spec("dinov2-large")["size"] == hub["size"]
    assert bb.preprocess_spec("dinov2-large")["crop_size"] == hub["crop_size"]
    a, b = bb.preprocess_spec("dinov2-large"), bb.preprocess_spec("dinov2-large", 448)
    assert {k for k in a if a[k] != b[k]} == {"size", "crop_size"}
    assert b["size"] == {"shortest_edge": 512} and b["crop_size"] == {"height": 448, "width": 448}


def test_backbone_settings_records_the_size_used():
    assert bb.backbone_settings("dinov2-large")["tokens_per_image"] == 256
    s = bb.backbone_settings("dinov2-large", input_res=448)
    assert s["tokens_per_image"] == 1024 and s["preprocess"]["crop_size"] == {"height": 448, "width": 448}
    assert s["preprocess_sha256"] == bb.preprocess_sha256("dinov2-large", 448)
    assert bb.backbone_settings("dinov2-large", input_res=224) == bb.backbone_settings("dinov2-large")


# ---------------------------------------------------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def run(monkeypatch, tmp_path):
    """``main`` up to its first heavy import: ImportError of the blocked ``datasets`` means "accepted"."""
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


def test_a_non_default_input_res_needs_an_output_dir(run, capsys):
    refused(run, capsys, "--input-res", "448", match="needs --output-dir")


@pytest.mark.parametrize("other", [["--backbone", "dinov2-reg-large"], ["--context-arm", "side_info"],
                                   ["--region-conditioning", "country", "--cities-json", "{cities}"],
                                   ["--init-from", "{init}"]])
def test_a_non_default_input_res_is_refused_with_another_experiment(run, capsys, other):
    refused(run, capsys, "--input-res", "518", "--output-dir", "o", *other, match="--input-res is compared")


def test_an_unregistered_size_is_refused(run, capsys):
    refused(run, capsys, "--input-res", "336", "--output-dir", "o", match="invalid choice")


def test_allowed_resolution_combinations_parse(run):
    assert run("--input-res", "448", "--output-dir", "o") == "accepted"
    assert run("--input-res", "518", "--output-dir", "o", "--batch-size", "4", "--gradient-checkpointing") == "accepted"
    assert run("--input-res", "448", "--output-dir", "o", "--target", "soft", "--split", "{split}") == "accepted"
    assert run("--input-res", "224", "--backbone", "convnextv2-base", "--output-dir", "o") == "accepted"
    assert run() == "accepted"   # the default run is unchanged


# ---------------------------------------------------------------------------------------------------------------------
# torch: the transforms, the saved processor, and DINOv2's position embeddings
# ---------------------------------------------------------------------------------------------------------------------

def _processor(res):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BitImageProcessor
    hub = json.load(open(os.path.join(FIX, "dinov2_large_preprocessor_config.json")))
    if res == 224:   # what AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=False) builds
        keep = {k: v for k, v in hub.items() if k not in ("image_processor_type", "_processor_class")}
        return BitImageProcessor(**keep)
    return BitImageProcessor(**bb.preprocess_spec("dinov2-large", res))


def _image(w=640, h=480):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_the_train_transform_is_the_inline_one_it_replaced():
    torch = pytest.importorskip("torch")
    from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
    p = _processor(224)
    crop = p.crop_size
    size = (crop["height"], crop["width"]) if "height" in crop else crop["shortest_edge"]
    old = Compose([RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True), RandomHorizontalFlip(), ToTensor(),
                   Normalize(mean=p.image_mean, std=p.image_std)])   # verbatim from fine_tune_classifier.py before
    new = ftc.build_train_transform(p)
    assert repr(new) == repr(old)
    img = _image()
    for seed in range(5):
        torch.manual_seed(seed)
        a = old(img)
        torch.manual_seed(seed)
        assert torch.equal(new(img), a)
    assert tuple(a.shape) == (3, 224, 224)


@pytest.mark.parametrize("res", [224, 448, 518])
def test_train_val_and_the_saved_processor_agree_on_the_size(res, tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import AutoImageProcessor
    p = _processor(res)
    torch.manual_seed(0)
    assert tuple(ftc.build_train_transform(p)(_image()).shape) == (3, res, res)
    val = p(_image(), return_tensors="pt")["pixel_values"]
    assert tuple(val.shape) == (1, 3, res, res)
    p.save_pretrained(tmp_path)   # the Trainer saves processing_class into every checkpoint; evaluate.py reads it
    q = AutoImageProcessor.from_pretrained(tmp_path, use_fast=False)
    assert type(q).__name__ == "BitImageProcessor"
    assert q.crop_size == {"height": res, "width": res} and q.size == {"shortest_edge": bb.input_sizes("dinov2-large", res)[0]}
    assert torch.equal(q(_image(), return_tensors="pt")["pixel_values"], val)


def test_the_default_size_processor_from_the_spec_equals_the_hub_one():
    torch = pytest.importorskip("torch")
    a = _processor(224)(_image(), return_tensors="pt")["pixel_values"]
    from transformers import BitImageProcessor
    b = BitImageProcessor(**bb.preprocess_spec("dinov2-large"))(_image(), return_tensors="pt")["pixel_values"]
    assert torch.equal(a, b)


def _tiny_dinov2():
    """A 2-layer, 32-wide DINOv2 with DINOv2-L's real geometry: patch 14 and a position table for 518 px (37 x 37),
    the grid facebook/dinov2-large ships (``image_size`` 518 in its config.json)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Dinov2Config, Dinov2ForImageClassification
    torch.manual_seed(0)
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       patch_size=14, image_size=518, num_labels=2)
    return Dinov2ForImageClassification(cfg).eval()


@pytest.mark.parametrize("res", [224, 448, 518])
def test_dinov2_interpolates_its_position_embeddings_to_every_study_size(res):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    model = _tiny_dinov2()
    emb = model.dinov2.embeddings
    table = emb.position_embeddings                       # 1 x (1 + 37*37) x 32
    assert table.shape[1] == 1 + 37 * 37
    x = torch.randn(1, 3, res, res)
    with torch.no_grad():
        tokens = emb(x)
        pos = emb.interpolate_pos_encoding(tokens, res, res)
        out = model(pixel_values=x)
    side = res // 14
    assert tokens.shape[1] == 1 + side * side and pos.shape[1] == 1 + side * side
    if res == 518:
        assert torch.equal(pos, table)                    # the pretraining grid: no interpolation at all
    else:
        grid = table[:, 1:].reshape(1, 37, 37, -1).permute(0, 3, 1, 2)
        want = F.interpolate(grid, size=(side, side), mode="bicubic", align_corners=False)
        assert torch.allclose(pos[:, 1:], want.permute(0, 2, 3, 1).reshape(1, side * side, -1), atol=1e-6)
        assert torch.equal(pos[:, :1], table[:, :1])      # the CLS position is never interpolated
    assert out.logits.shape == (1, 2) and torch.isfinite(out.logits).all()


def test_a_448_step_trains_the_position_table_under_gradient_checkpointing():
    torch = pytest.importorskip("torch")
    model = _tiny_dinov2().train()
    assert model.supports_gradient_checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    x, y = torch.randn(2, 3, 448, 448), torch.tensor([0, 1])
    loss = model(pixel_values=x, labels=y).loss
    loss.backward()
    g = model.dinov2.embeddings.position_embeddings.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # checkpointing recomputes activations; it does not change the gradient (DINOv2 has no dropout)
    ref = _tiny_dinov2().train()
    ref.load_state_dict(model.state_dict())
    ref(pixel_values=x, labels=y).loss.backward()
    assert torch.allclose(ref.dinov2.embeddings.position_embeddings.grad, g, atol=1e-6)
