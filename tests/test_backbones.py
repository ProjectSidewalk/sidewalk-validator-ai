"""backbones.py (issue #119): the registry, the fixed preprocessing, the RampNet key map, prefetch, and (torch only)
every family's classifier under the Trainer with gradient accumulation.

The RampNet name list is the real one from the released ``projectsidewalk/rampnet-model`` safetensors header
(``tests/fixtures/backbones/rampnet_model_keys.json``: name -> shape, metadata and offsets dropped). The template list
was generated once from ``ConvNextV2Config`` at base dims with random init, no download::

    cfg = transformers.ConvNextV2Config(depths=[3, 3, 27, 3], hidden_sizes=[128, 256, 512, 1024], num_labels=2)
    with torch.device("meta"):
        m = transformers.ConvNextV2ForImageClassification(cfg)
    {k: list(v.shape) for k, v in sorted(m.state_dict().items()) if k.startswith("convnextv2.")}
"""
import json
import os
import re

import pytest

import backbones as bb

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "backbones")
ID2LABEL = {0: "correct", 1: "incorrect"}
LABEL2ID = {"correct": 0, "incorrect": 1}


def fixture(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)


def tag_ok(tag):
    """evaluate.py's --tag rule."""
    return tag.replace("-", "").replace("_", "").isalnum()


# ---------------------------------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------------------------------

def test_registry_rows_are_well_formed():
    import fine_tune_classifier as ftc
    assert len(bb.REGISTRY) == len(bb._ROWS) == 9   # unique keys
    assert bb.REGISTRY[bb.DEFAULT].source == ftc.CHECKPOINT
    assert set(bb.ELIGIBLE) <= set(bb.REGISTRY) and bb.DEFAULT not in bb.ELIGIBLE
    for key, b in bb.REGISTRY.items():
        assert not re.search(r"-s\d+$", key), key   # a tag's -s<seed> suffix must stay unambiguous
        for tag in (f"bb-{key}", f"bb-{key}-s43", f"bb-{key}-lr3e5", f"smoke-{key}", f"smoke-rand-{key}"):
            assert tag_ok(tag), tag
        assert re.fullmatch(r"[0-9a-f]{40}", b.revision), key
        assert b.family in bb.POOLING and b.family in bb.MODEL_CLASS
        if b.patch_size:
            assert b.crop_px % b.patch_size == 0, key
        assert b.arch, key   # every row can be built with random weights for a smoke test
    tokens = {k: bb.get(k).tokens_per_image for k in ("dinov2-large", "dinov2-reg-large", "dinov3-vitl16",
                                                       "dinov3-vitl16-256", "convnextv2-base")}
    assert tokens == {"dinov2-large": 256, "dinov2-reg-large": 256, "dinov3-vitl16": 196, "dinov3-vitl16-256": 256,
                      "convnextv2-base": 49}
    assert [k for k, b in bb.REGISTRY.items() if b.gated] == ["dinov3-vitl16", "dinov3-vitl16-256", "dinov3-vitb16",
                                                             "dinov3-vits16"]
    with pytest.raises(KeyError, match="unknown backbone"):
        bb.get("dinov4")


def test_the_default_preprocessing_equals_the_hub_config():
    hub = fixture("dinov2_large_preprocessor_config.json")
    spec = bb.preprocess_spec(bb.DEFAULT)
    shared = set(hub) & set(spec)
    assert shared >= {"size", "crop_size", "resample", "rescale_factor", "image_mean", "image_std", "do_resize",
                      "do_center_crop", "do_rescale", "do_normalize", "do_convert_rgb"}
    for k in shared:
        assert spec[k] == pytest.approx(hub[k]) if isinstance(hub[k], float) else spec[k] == hub[k], k
    # every arm is preprocessed alike, except the token-matched DINOv3 row
    assert {bb.preprocess_sha256(k) for k in bb.REGISTRY if k != "dinov3-vitl16-256"} == {bb.preprocess_sha256(bb.DEFAULT)}
    assert bb.preprocess_spec("dinov3-vitl16-256")["crop_size"] == {"height": 256, "width": 256}


def test_preprocess_sha256_is_stable():
    assert bb.preprocess_sha256("dinov2-large") == bb.preprocess_sha256(bb.get("dinov2-large"))
    assert re.fullmatch(r"[0-9a-f]{64}", bb.preprocess_sha256("dinov2-large"))


def test_backbone_settings_shape_without_a_model():
    s = bb.backbone_settings("dinov3-vitl16")
    assert set(s) == {"key", "source", "revision", "resolved_revision", "family", "model_class", "pooling",
                      "patch_size", "num_prefix_tokens", "tokens_per_image", "params", "accepts_loss_kwargs",
                      "random_init", "pos_embed", "preprocess", "preprocess_sha256", "weights_sha256", "conversion"}
    assert s["model_class"] == "SidewalkDinov3ForImageClassification" and s["num_prefix_tokens"] == 5
    assert s["preprocess_sha256"] == bb.preprocess_sha256("dinov3-vitl16")


def test_converted_rows_need_the_conversion_output(tmp_path, monkeypatch):
    monkeypatch.delenv(bb.BACKBONE_DIR_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="--backbone-dir"):
        bb.resolve_source("rampnet-convnextv2-base")
    with pytest.raises(FileNotFoundError, match="convert-rampnet"):
        bb.resolve_source("rampnet-convnextv2-base", str(tmp_path))
    d = tmp_path / "rampnet-convnextv2-base"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "conversion.json").write_text("{}")
    assert bb.resolve_source("rampnet-convnextv2-base", str(tmp_path)) == (str(d), None)
    monkeypatch.setenv(bb.BACKBONE_DIR_ENV, str(tmp_path))
    assert bb.resolve_source("rampnet-convnextv2-base") == (str(d), None)
    assert bb.resolve_source("convnextv2-base") == ("facebook/convnextv2-base-22k-384", bb.get("convnextv2-base").revision)


# ---------------------------------------------------------------------------------------------------------------------
# RampNet key map on the real names
# ---------------------------------------------------------------------------------------------------------------------

def test_rampnet_map_is_a_bijection_onto_the_template_with_equal_shapes():
    src, template = fixture("rampnet_model_keys.json"), fixture("convnextv2_base_keys.json")
    assert len(src) == 380 and sum(k.startswith("model.head.") for k in src) == 4
    mapping = bb.plan_conversion(src, template)
    assert len(mapping) == 376 == len(set(mapping.values()))
    covered = {k for k in template if k.startswith(("convnextv2.embeddings.", "convnextv2.encoder."))}
    assert set(mapping.values()) == covered
    assert set(template) - covered == {"convnextv2.layernorm.weight", "convnextv2.layernorm.bias"}
    grn = [k for k in mapping if ".grn." in k]
    assert grn and all(len(src[k]) == 1 and template[mapping[k]][:3] == [1, 1, 1] for k in grn)


def test_rampnet_map_accepts_names_with_and_without_the_model_prefix():
    src = fixture("rampnet_model_keys.json")
    bare = {k[len("model."):]: v for k, v in src.items()}
    assert bb.plan_conversion(bare, fixture("convnextv2_base_keys.json"))


def test_rampnet_map_raises_on_an_unknown_or_a_missing_name():
    src, template = fixture("rampnet_model_keys.json"), fixture("convnextv2_base_keys.json")
    with pytest.raises(ValueError, match="cannot place"):
        bb.rampnet_key_map([*src, "model.feature_extractor.1.0.blocks.0.gamma"])
    with pytest.raises(ValueError, match="cannot place"):
        bb.rampnet_key_map(["model.backbone.stem.0.weight"])
    missing = dict(src)
    missing.pop("model.feature_extractor.1.3.blocks.2.mlp.grn.weight")
    with pytest.raises(ValueError, match="not covered"):
        bb.plan_conversion(missing, template)
    wrong = {**src, "model.feature_extractor.0.0.weight": [64, 3, 4, 4]}
    with pytest.raises(ValueError, match="shape mismatches"):
        bb.plan_conversion(wrong, template)


# ---------------------------------------------------------------------------------------------------------------------
# prefetch and the storage check (no huggingface_hub needed)
# ---------------------------------------------------------------------------------------------------------------------

class GatedRepoError(Exception):
    pass


class HTTPError(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.response = type("R", (), {"status_code": status})()


def _fake_hub(tmp_path, fail=None):
    """A download() that writes a snapshot dir per repo; ``fail`` maps a repo id to the exception it raises."""
    calls = []

    def download(repo_id, revision, allow_patterns):
        calls.append(repo_id)
        if fail and repo_id in fail:
            raise fail[repo_id]
        repo = tmp_path / ("models--" + repo_id.replace("/", "--"))
        snap = repo / "snapshots" / revision
        snap.mkdir(parents=True, exist_ok=True)
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / "main").write_text(revision)
        for name in allow_patterns:
            if name != "preprocessor_config.json":
                (snap / name).write_bytes(f"{repo_id}:{name}".encode())
        return str(snap)
    return download, calls


def test_prefetch_skips_gated_rows_without_a_token_and_records_files(tmp_path):
    download, calls = _fake_hub(tmp_path)
    m = bb.prefetch(["dinov2-large", "convnextv2-base", "dinov3-vitl16"], download=download, env={})
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" not in calls      # not attempted
    assert m["dinov3-vitl16"] == {"source": bb.get("dinov3-vitl16").source, "revision": bb.get("dinov3-vitl16").revision,
                                  "gated": True, "available": False, "reason": bb.GATED_REASON}
    assert m["convnextv2-base"]["available"] and set(m["convnextv2-base"]["files"]) == {"config.json", "model.safetensors"}
    assert re.fullmatch(r"[0-9a-f]{64}", m["convnextv2-base"]["files"]["model.safetensors"]["sha256"])
    assert m["dinov2-large"]["refs_main"] == bb.get("dinov2-large").revision


def test_prefetch_one_failure_leaves_the_others_and_never_leaks_the_token(tmp_path, monkeypatch, capsys):
    token = "hf_THISISNOTAREALTOKEN123"
    fail = {"facebook/dinov2-with-registers-large": HTTPError(f"500 for url ...?token={token}", 500),
            "facebook/dinov3-vitl16-pretrain-lvd1689m": GatedRepoError("Access to model is restricted")}
    download, calls = _fake_hub(tmp_path, fail)
    m = bb.prefetch(["dinov2-reg-large", "convnextv2-base", "dinov3-vitl16"], download=download,
                    env={"HF_TOKEN": token})
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" in calls        # attempted with a token
    assert m["dinov3-vitl16"]["reason"] == bb.GATED_REASON
    assert not m["dinov2-reg-large"]["available"] and "HTTP 500" in m["dinov2-reg-large"]["reason"]
    assert m["convnextv2-base"]["available"]
    assert token not in json.dumps(m)
    # the CLI: exit 3 when any row is unavailable, the token nowhere in its output or manifest
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(bb, "_default_download", download)
    out = tmp_path / "manifest.json"
    assert bb.main(["prefetch", "--backbones", "dinov2-reg-large,convnextv2-base", "--out", str(out)]) == 3
    io = capsys.readouterr()
    assert token not in io.out + io.err + out.read_text()
    assert bb.main(["prefetch", "--backbones", "convnextv2-base", "--out", str(out)]) == 0


def test_prefetch_refuses_a_default_whose_refs_main_moved(tmp_path):
    download, _ = _fake_hub(tmp_path)

    def moved(repo_id, revision, allow_patterns):
        snap = download(repo_id, revision, allow_patterns)
        (tmp_path / ("models--" + repo_id.replace("/", "--")) / "refs" / "main").write_text("f" * 40)
        return snap
    m = bb.prefetch(["dinov2-large"], download=moved, env={})
    assert not m["dinov2-large"]["available"] and "refs/main" in m["dinov2-large"]["reason"]


def test_storage_free_reads_hyakstorage(monkeypatch, capsys):
    import io
    text = ("│ Total:            │ 969GB / 1024GB             │ 974033 / 1000000 files      │\n"
            "│                   │ 95%                        │ 97%                         │\n")
    assert bb.parse_hyakstorage(text) == (55, 25967)
    assert bb.parse_hyakstorage("│ Total: │ 1.5TB / 2TB │ 10 / 100 files │") == (512, 90)
    assert bb.parse_hyakstorage("nothing here") is None
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    assert bb.main(["storage-free", "--min-gb", "10", "--min-files", "2000"]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    assert bb.main(["storage-free", "--min-gb", "60", "--min-files", "2000"]) == 4


def test_list_prints_every_row(capsys):
    assert bb.main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(f"`{k}`" in out for k in bb.REGISTRY)


# ---------------------------------------------------------------------------------------------------------------------
# torch only: every family under the Trainer (skipped in CI; run locally and in the klone smoke job)
# ---------------------------------------------------------------------------------------------------------------------

# Tiny configs, every dropout and drop-path rate 0, so a train-mode forward is deterministic.
TINY = {
    "dinov2-large": ({"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "mlp_ratio": 2,
                      "image_size": 28, "patch_size": 14, "hidden_dropout_prob": 0.0,
                      "attention_probs_dropout_prob": 0.0, "drop_path_rate": 0.0}, 28),
    "dinov2-reg-large": ({"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "mlp_ratio": 2,
                          "image_size": 28, "patch_size": 14, "num_register_tokens": 4, "hidden_dropout_prob": 0.0,
                          "attention_probs_dropout_prob": 0.0, "drop_path_rate": 0.0}, 28),
    "convnextv2-base": ({"depths": [1, 1, 1, 1], "hidden_sizes": [8, 16, 32, 64], "drop_path_rate": 0.0}, 64),
    "dinov3-vitl16": ({"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
                       "num_register_tokens": 4, "patch_size": 16, "image_size": 32, "layerscale_value": 1e-5,
                       "attention_dropout": 0.0, "drop_path_rate": 0.0}, 32),
}


def tiny(key, pos_embed_aug="off"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if bb.get(key).family == "dinov3_vit" and not bb.has_dinov3():
        pytest.skip("this transformers has no DINOv3 (< 4.56)")
    torch.manual_seed(0)
    arch, size = TINY[key]
    model = bb.build_classifier(key, ID2LABEL, LABEL2ID, random_init=True, arch=arch, pos_embed_aug=pos_embed_aug)
    return model, size


class _Rows:
    """A tiny map-style dataset of fixed images and labels (and soft targets)."""

    def __init__(self, n, size, soft=False):
        import torch
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 3, size, size, generator=g)
        self.y = torch.tensor([i % 2 for i in range(n)])
        self.p = torch.linspace(0.1, 0.9, n) if soft else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        row = {"pixel_values": self.x[i], "labels": self.y[i]}
        if self.p is not None:
            row["soft_targets"] = self.p[i]
        return row


def _trainer(model, data, tmp_path, batch, accum, trainer_cls=None):
    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(output_dir=str(tmp_path / f"t{batch}x{accum}"), per_device_train_batch_size=batch,
                             gradient_accumulation_steps=accum, max_steps=1, learning_rate=1e-3,
                             logging_steps=1, report_to="none", use_cpu=True, save_strategy="no", seed=0,
                             max_grad_norm=1e9, remove_unused_columns=False, dataloader_num_workers=0)
    return (trainer_cls or Trainer)(model=model, args=args, train_dataset=data)


@pytest.mark.parametrize("key", list(TINY))
def test_every_family_trains_with_accumulation_and_divides_by_the_steps(key, tmp_path):
    model, size = tiny(key)
    assert model.accepts_loss_kwargs is False and model._sva_random_init
    trainer = _trainer(model, _Rows(8, size), tmp_path, 2, 4)
    assert trainer.model_accepts_loss_kwargs is False
    trainer.train()
    assert trainer.state.global_step == 1


@pytest.mark.parametrize("key", list(TINY))
def test_grad_norm_at_2x4_equals_8x1(key, tmp_path):
    """Loss divided by the accumulation steps: batch 2 x 4 and batch 8 x 1 give the same gradient on the same rows
    (a model whose loss is not divided would give 4x the norm)."""
    import copy
    model, size = tiny(key)
    data = _Rows(8, size)
    norms = []
    for batch, accum in ((2, 4), (8, 1)):
        t = _trainer(copy.deepcopy(model), data, tmp_path, batch, accum)
        t.train()
        norms.append(next(h["grad_norm"] for h in t.state.log_history if "grad_norm" in h))
    assert norms[0] == pytest.approx(norms[1], rel=1e-4)


@pytest.mark.parametrize("key", list(TINY))
def test_soft_target_trainer_step_on_every_family(key, tmp_path):
    from transformers import Trainer

    import fine_tune_classifier as ftc
    model, size = tiny(key)
    t = _trainer(model, _Rows(8, size, soft=True), tmp_path, 2, 4, ftc.soft_target_trainer(Trainer))
    assert t.model_accepts_loss_kwargs is False
    t.train()
    assert t.state.global_step == 1


def test_dinov3_wrapper_pools_cls_and_patches_and_round_trips_through_auto(tmp_path):
    import torch
    from transformers import AutoModelForImageClassification
    model, size = tiny("dinov3-vitl16")
    assert model.base_model is model.dinov3_vit and type(model).__name__ == "SidewalkDinov3ForImageClassification"
    x = torch.randn(3, 3, size, size, generator=torch.Generator().manual_seed(2))
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=x, labels=torch.tensor([0, 1, 0]))
    assert out.logits.shape == (3, 2) and out.loss.ndim == 0 and torch.isfinite(out.loss)

    # register tokens never reach the head: perturb them after the encoder
    def perturb(_module, _inputs, output):
        h = output.last_hidden_state.clone()
        h[:, 1:5] += 100.0
        output.last_hidden_state = h
        return output
    handle = model.dinov3_vit.register_forward_hook(perturb)
    with torch.no_grad():
        perturbed = model(pixel_values=x).logits
    handle.remove()
    assert torch.allclose(perturbed, out.logits, atol=1e-6)

    model.save_pretrained(tmp_path / "ck")
    assert json.loads((tmp_path / "ck" / "config.json").read_text())["model_type"] == "sidewalk_dinov3_classifier"
    assert bb.register_auto() and bb.register_auto()   # idempotent
    loaded = AutoModelForImageClassification.from_pretrained(tmp_path / "ck").eval()
    with torch.no_grad():
        assert torch.equal(loaded(pixel_values=x).logits, out.logits)


def test_dinov3_backbone_weights_load_into_the_wrapper(tmp_path):
    """A bare DINOv3ViTModel checkpoint (the hub layout) loads into the wrapper through base_model_prefix."""
    import torch
    tiny("dinov3-vitl16")   # skips without DINOv3
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    arch = {k: v for k, v in TINY["dinov3-vitl16"][0].items()}
    base = DINOv3ViTModel(DINOv3ViTConfig(**arch))
    base.save_pretrained(tmp_path / "hub")
    _, cls = bb._dinov3_classes()
    m = cls.from_pretrained(tmp_path / "hub", num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    assert torch.equal(m.dinov3_vit.embeddings.cls_token, base.embeddings.cls_token)
    assert torch.equal(m.dinov3_vit.layer[1].mlp.up_proj.weight, base.layer[1].mlp.up_proj.weight)


def test_dinov3_positional_augmentation_switch():
    import torch
    x = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(3))
    off, _ = tiny("dinov3-vitl16", pos_embed_aug="off")
    off.train()
    with torch.no_grad():
        assert torch.equal(off(pixel_values=x).logits, off(pixel_values=x).logits)
    assert bb.backbone_settings("dinov3-vitl16", off)["pos_embed"] == {"shift": None, "jitter": None, "rescale": None}
    on, _ = tiny("dinov3-vitl16", pos_embed_aug="as_released")   # the library default: rescale 2.0
    assert on.config.pos_embed_rescale == 2.0
    on.train()
    torch.manual_seed(5)
    with torch.no_grad():
        assert not torch.equal(on(pixel_values=x).logits, on(pixel_values=x).logits)


def test_backbone_settings_records_the_built_model():
    model, _ = tiny("convnextv2-base")
    s = bb.backbone_settings("convnextv2-base", model)
    assert s["params"] == sum(p.numel() for p in model.parameters()) and s["accepts_loss_kwargs"] is False
    assert s["random_init"] is True and s["weights_sha256"] is None and s["model_class"] == "ConvNextV2ForImageClassification"


def test_full_size_random_dinov3_has_the_registry_parameter_count():
    pytest.importorskip("torch")
    if not bb.has_dinov3():
        pytest.skip("this transformers has no DINOv3 (< 4.56)")
    import torch
    cfg_cls, _ = bb._dinov3_classes()
    from transformers import DINOv3ViTModel
    with torch.device("meta"):
        enc = DINOv3ViTModel(cfg_cls(**bb.get("dinov3-vitl16").arch))
    assert sum(p.numel() for p in enc.parameters()) / 1e6 == pytest.approx(303.1, rel=0.01)


def test_saved_processor_reloads_as_the_slow_bit_processor(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoImageProcessor, BitImageProcessor
    p = bb.build_image_processor("convnextv2-base")
    assert isinstance(p, BitImageProcessor)
    p.save_pretrained(tmp_path)
    q = AutoImageProcessor.from_pretrained(tmp_path, use_fast=False)
    assert type(q).__name__ == "BitImageProcessor"
    assert q.size == {"shortest_edge": 256} and q.crop_size == {"height": 224, "width": 224} and q.resample == 3


def test_timm_parity_of_the_key_map_on_a_tiny_convnext():
    """convnextv2_atto (random) as RampNet's feature_extractor -> key map -> ConvNextV2 at atto dims: same features."""
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from torch import nn
    from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
    torch.manual_seed(0)
    net = timm.create_model("convnextv2_atto", pretrained=False, conv_mlp=False)   # Linear MLP, as convnextv2_base
    fe = nn.Sequential(*list(net.children())[:-2])
    src = {f"model.feature_extractor.{k}": v for k, v in fe.state_dict().items()}
    src["model.head.0.weight"] = torch.zeros(1)   # dropped by the map
    hf = ConvNextV2ForImageClassification(ConvNextV2Config(depths=[2, 2, 6, 2], hidden_sizes=[40, 80, 160, 320],
                                                           num_labels=2))
    sd, counts = bb.convert_state_dict(src, hf.state_dict())
    hf.load_state_dict(sd, strict=True)
    assert counts["from_rampnet"] == len(src) - 1 and counts["kept_from_template"] == 2
    assert bb.timm_parity(src, hf, "convnextv2_atto", size=64, conv_mlp=False) < 1e-5


REG_ONNX = ("Dinov2WithRegisters interpolates its position embeddings with antialiased bicubic upsampling, which "
            "torch.onnx cannot export at opset 17 (known limit, docs/backbone-2026-11.md)")


@pytest.mark.parametrize("key", ["convnextv2-base", "dinov3-vitl16",
                                 pytest.param("dinov2-reg-large", marks=pytest.mark.xfail(reason=REG_ONNX))])
def test_onnx_export_of_each_family(key, tmp_path):
    """export_onnx.export_model on a tiny checkpoint of each new family agrees with PyTorch (a failure is recorded in
    docs/backbone-2026-11.md as a known limit of the later ONNX stage, not a blocker for the pilot)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from transformers import BitImageProcessor

    import export_onnx
    model, size = tiny(key)
    model.eval()
    ck = tmp_path / "ck"
    model.save_pretrained(ck)
    BitImageProcessor(size={"shortest_edge": size}, crop_size={"height": size, "width": size}).save_pretrained(ck)
    path = export_onnx.export_model(str(ck), str(tmp_path / "onnx"))
    x = torch.randn(2, 3, size, size, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        ref = model(pixel_values=x).logits.numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    assert abs(got - ref).max() < 1e-4
