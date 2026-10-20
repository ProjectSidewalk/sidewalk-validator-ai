"""Tests for explain.py: rollout, Grad-CAM, categories and sampling, rendering, coding sheet, kappa.

The model tests use a tiny random Dinov2ForImageClassification (2 layers, 56 px images = 4 x 4 patches of 14 px), so
nothing is downloaded; they are skipped when torch/transformers are not installed (CI installs neither).
"""
import numpy as np
import pandas as pd
import pytest

import explain


# ---------------------------------------------------------------------------------------------------------------------
# Rollout (numpy only)
# ---------------------------------------------------------------------------------------------------------------------

def random_attention(rng, heads, t):
    a = rng.random((heads, t, t))
    return a / a.sum(-1, keepdims=True)


def test_rollout_shape_and_cls_row_sums_to_one():
    rng = np.random.default_rng(0)
    layers = [random_attention(rng, 3, 17) for _ in range(4)]
    r = explain.attention_rollout(layers)
    assert r.shape == (16,)
    assert (r >= 0).all()
    # The rolled-out matrix is a product of row-stochastic matrices, so the full CLS row (CLS + patches) sums to 1;
    # the patch part is that minus what CLS keeps for itself, strictly between 0 and 1.
    assert 0 < r.sum() < 1


def test_rollout_identity_attention_keeps_cls_on_itself():
    eye = np.eye(5)[None].repeat(2, axis=0)
    np.testing.assert_allclose(explain.attention_rollout([eye, eye, eye]), 0.0)


def test_rollout_uniform_attention_is_uniform_over_patches():
    t = 5
    uniform = np.full((2, t, t), 1 / t)
    r = explain.attention_rollout([uniform, uniform])
    np.testing.assert_allclose(r, r[0])
    # one layer by hand: 0.5 * 1/5 + 0.5 * I, CLS row = [0.6, 0.1, 0.1, 0.1, 0.1]
    np.testing.assert_allclose(explain.attention_rollout([uniform]), [0.1] * 4)


def test_rollout_accepts_batch_dim_and_head_fusions():
    rng = np.random.default_rng(1)
    layers = [random_attention(rng, 2, 10) for _ in range(2)]
    batched = [a[None] for a in layers]
    np.testing.assert_allclose(explain.attention_rollout(layers), explain.attention_rollout(batched))
    for fusion in ("max", "min"):
        assert explain.attention_rollout(layers, head_fusion=fusion).shape == (9,)
    discarded = explain.attention_rollout(layers, discard_ratio=0.9)
    assert discarded.shape == (9,) and np.isfinite(discarded).all()


def test_last_layer_cls_attention_and_grid():
    rng = np.random.default_rng(2)
    layers = [random_attention(rng, 3, 17) for _ in range(2)]
    cls = explain.last_layer_cls_attention(layers)
    assert cls.shape == (3, 16)
    np.testing.assert_allclose(cls, layers[-1][:, 0, 1:])
    assert explain.to_grid(np.arange(16)).shape == (4, 4)
    with pytest.raises(ValueError):
        explain.to_grid(np.arange(15))


def test_centre_and_top_mass():
    g = np.zeros((16, 16))
    g[4:12, 4:12] = 1
    assert explain.centre_mass(g) == 1.0
    assert explain.centre_mass(np.ones((16, 16))) == 0.25
    assert np.isnan(explain.centre_mass(np.zeros((4, 4))))
    assert explain.top_mass([0, 0, 5, 5], k=2) == 1.0


# ---------------------------------------------------------------------------------------------------------------------
# Categories and sampling
# ---------------------------------------------------------------------------------------------------------------------

def synthetic_predictions(n=200, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.choice(["correct", "incorrect"], n)
    p_correct = rng.random(n)
    return pd.DataFrame({"label_uid": [f"city{i % 3}:{i}" for i in range(n)], "truth": truth,
                         "p_correct": p_correct, "p_incorrect": 1 - p_correct})


def test_categorise_against_the_positive_class():
    assert explain.categorise(["correct", "incorrect", "correct", "incorrect"],
                              ["correct", "correct", "incorrect", "incorrect"]) == ["TP", "FP", "FN", "TN"]
    with pytest.raises(ValueError):
        explain.categorise(["agree"], ["correct"])


def test_add_verdicts_uses_the_api_tie_rule():
    df = pd.DataFrame({"label_uid": ["a:1", "a:2"], "truth": ["incorrect", "correct"],
                       "p_correct": [0.5, 0.2], "p_incorrect": [0.5, 0.8]})
    out = explain.add_verdicts(df)
    assert out.pred.tolist() == ["correct", "incorrect"]        # a tie goes to index 0, correct
    assert out.category.tolist() == ["FP", "FN"]
    assert out.confidence.tolist() == [0.5, 0.8]


def test_sample_categories_caps_orders_and_is_deterministic():
    preds = explain.add_verdicts(synthetic_predictions())
    s1 = explain.sample_categories(preds, 10, seed=7, label_type="Obstacle")
    s2 = explain.sample_categories(preds.sample(frac=1, random_state=3), 10, seed=7, label_type="Obstacle")
    assert s1.label_uid.tolist() == s2.label_uid.tolist()                  # independent of input order
    assert set(s1.category) == {"TP", "FP", "FN"}
    assert (s1.groupby("category").size() == 10).all()
    for _, g in s1.groupby("category"):
        assert g.confidence.is_monotonic_decreasing
        assert g.tile.tolist() == list(range(1, 11))
    assert s1.overlay_id.iloc[0] == "obstacle-TP-01"
    assert s1.label_uid.is_unique


def test_sample_categories_keeps_everything_in_a_small_category():
    preds = explain.add_verdicts(synthetic_predictions())
    fn = int((preds.category == "FN").sum())
    s = explain.sample_categories(preds, fn + 5, seed=1)
    assert (s.category == "FN").sum() == fn


def test_load_crops_dir_builds_city_qualified_uids(tmp_path):
    for cls, name in (("correct", "new_taipei_12.webp"), ("incorrect", "sea_12.webp")):
        d = tmp_path / "test" / cls
        d.mkdir(parents=True)
        (d / name).write_bytes(b"x")
    m = explain.load_crops_dir(str(tmp_path))
    assert m.label_uid.tolist() == ["new_taipei:12", "sea:12"]
    assert m.truth.tolist() == ["correct", "incorrect"]
    with pytest.raises(FileNotFoundError):
        explain.load_crops_dir(str(tmp_path), split="val")


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------

def test_overlay_writer_and_contact_sheet_produce_files(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(3)
    image = (rng.random((56, 56, 3)) * 255).astype(np.uint8)
    grids = [rng.random((4, 4)), np.zeros((4, 4))]           # a constant map must not divide by zero
    path = tmp_path / "o.png"
    explain.write_overlay(str(path), image, grids, "x:1 truth correct pred correct")
    assert path.stat().st_size > 0
    s = explain.strip(image, grids)
    assert s.shape == (56, 168, 3) and s.dtype == np.uint8
    sheet = tmp_path / "sheet.jpg"
    explain.contact_sheet([s] * 5, [f"{i:02d}" for i in range(5)], str(sheet), ncols=2, tile_height=40, header="h")
    with Image.open(sheet) as im:
        assert im.format == "JPEG" and im.height > 3 * 40
    empty = tmp_path / "empty.jpg"
    explain.contact_sheet([], [], str(empty), header="none")
    assert empty.stat().st_size > 0


def test_denormalise_round_trip():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image = np.random.default_rng(4).integers(0, 256, (8, 8, 3)).astype(np.uint8)
    x = ((image / 255.0) - mean) / std
    np.testing.assert_array_equal(explain.denormalise(x.transpose(2, 0, 1), mean, std), image)


# ---------------------------------------------------------------------------------------------------------------------
# Coding sheet and agreement
# ---------------------------------------------------------------------------------------------------------------------

def filled(values, column="informative", types=None):
    n = len(values)
    return pd.DataFrame({"overlay_id": [f"o{i}" for i in range(n)], "label_type": types or ["X"] * n,
                         column: values})


def test_kappa_hand_computed():
    # p_o = 3/4; rater 1 says yes 2/4, rater 2 3/4, so p_e = 0.5*0.75 + 0.5*0.25 = 0.5; kappa = 0.25 / 0.5 = 0.5
    t = explain.agreement(filled(["yes", "yes", "no", "no"]), filled(["yes", "yes", "no", "yes"]), ["informative"])
    row = t[t.group == "all"].iloc[0]
    assert row.n == 4 and row.percent_agreement == 0.75 and row.kappa == pytest.approx(0.5)


def test_kappa_skips_blank_and_unclear_and_groups_by_type():
    a = filled(["yes", "partly", "no", "unclear", "yes", ""], "rollout_covers", ["A", "A", "A", "A", "B", "B"])
    b = filled(["yes", "no", "no", "yes", "yes", "no"], "rollout_covers", ["A", "A", "A", "A", "B", "B"])
    t = explain.agreement(a, b, ["rollout_covers"]).set_index("group")
    assert t.loc["all", "n"] == 4 and t.loc["A", "n"] == 3 and t.loc["B", "n"] == 1
    assert t.loc["all", "percent_agreement"] == 0.75
    assert np.isfinite(t.loc["all", "weighted_kappa"])
    assert t.loc["B", "kappa"] == 1.0                                      # one shared value: perfect agreement


def test_agreement_rejects_duplicate_ids_and_bad_codes():
    a = filled(["yes", "no"])
    with pytest.raises(pd.errors.MergeError):
        explain.agreement(pd.concat([a, a]), a, ["informative"])
    sheet = explain.coding_sheet(pd.DataFrame({c: ["v"] for c in explain.SHEET_ID_COLUMNS}))
    assert list(sheet.columns) == explain.SHEET_ID_COLUMNS + explain.SHEET_RATER_COLUMNS
    sheet.loc[0, "informative"] = "maybe"
    with pytest.raises(ValueError, match="informative"):
        explain.validate_codes(sheet)


# ---------------------------------------------------------------------------------------------------------------------
# Tiny DINOv2 classifier (torch + transformers)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2,
                                       image_size=56, patch_size=14, num_labels=2,
                                       id2label={0: "correct", 1: "incorrect"},
                                       label2id={"correct": 0, "incorrect": 1})
    model = transformers.Dinov2ForImageClassification._from_config(config, attn_implementation="eager").eval()
    return model


def test_explain_one_shapes_and_signs(tiny_model):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 3, 56, 56)
    r = explain.explain_one(tiny_model, x)
    assert r["rollout"].shape == (16,) and r["gradcam"].shape == (16,)
    assert r["cls_attention"].shape == (2, 16)
    assert (r["gradcam"] >= 0).all() and (r["rollout"] >= 0).all()
    with torch.no_grad():
        probs = tiny_model(pixel_values=x).logits.softmax(-1)[0].numpy()
    np.testing.assert_allclose(r["probs"], probs, rtol=1e-5)
    assert r["target"] == int(probs.argmax())
    assert explain.to_grid(r["gradcam"]).shape == (4, 4)
    assert all(p.grad is None for p in tiny_model.parameters())               # no gradients left behind


def test_grad_cam_targets_differ_and_hook_is_removed(tiny_model):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 3, 56, 56)
    cam0, t0 = explain.grad_cam(tiny_model, x, target_class=0)
    cam1, t1 = explain.grad_cam(tiny_model, x, target_class=1)
    assert (t0, t1) == (0, 1)
    # For two classes the margins are negatives of each other, so the pre-ReLU maps are too: where one is positive
    # the other is zero.
    assert not np.any((cam0 > 1e-6) & (cam1 > 1e-6))
    assert len(tiny_model.dinov2.encoder.layer[-1].norm1._forward_hooks) == 0


def test_explain_one_needs_eager_attention():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2,
                                       image_size=28, patch_size=14, num_labels=2)
    model = transformers.Dinov2ForImageClassification._from_config(config, attn_implementation="sdpa").eval()
    # transformers 4.55 falls back to eager when attentions are requested, so this still works; the check guards
    # against a future version returning None instead.
    r = explain.explain_one(model, torch.randn(1, 3, 28, 28))
    assert r["rollout"].shape == (4,)


def test_city_ids_keeps_public_deployments_only():
    cities = {"cities": [{"city_id": "chicago-il", "visibility": "public"},
                         {"city_id": "seattle-wa", "visibility": "public"},
                         {"city_id": "st-louis-mo", "visibility": "public"},
                         {"city_id": "new-taipei-tw", "visibility": "private"},
                         {"city_id": "taipei", "visibility": "private"}]}
    got = explain.city_ids(["chicago", "sea", "st_louis", "new_taipei", "taipei", "atlantis"], cities)
    assert got == {"chicago": "chicago-il", "sea": "seattle-wa", "st_louis": "st-louis-mo"}


def test_summarise_medians_per_type_and_category():
    s = pd.DataFrame({"label_type": ["A", "A", "A", "B"], "category": ["TP", "TP", "FP", "TP"],
                      "confidence": [0.9, 0.7, 0.6, 0.99], "rollout_centre_mass": [0.2, 0.4, 0.1, 0.5],
                      "gradcam_centre_mass": [0.3, 0.5, 0.2, 0.6]})
    t = explain.summarise(s).set_index(["label_type", "category"])
    assert t.loc[("A", "TP"), "n"] == 2
    assert t.loc[("A", "TP"), "median_confidence"] == pytest.approx(0.8)
    assert t.loc[("A", "TP"), "median_rollout_centre_mass"] == pytest.approx(0.3)
    assert "median_rollout_top3_mass" not in t.columns
