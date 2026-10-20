"""Tests for explain.py: maps, statistics, categories and sampling, deployments, rendering, packet, kappa, go/no-go.

The model tests use a tiny random Dinov2ForImageClassification (2 layers, 56 px images = 4 x 4 patches of 14 px), so
nothing is downloaded; they are skipped when torch/transformers are not installed (CI installs neither). Deployment
ids in this file are fictitious.
"""
import types

import numpy as np
import pandas as pd
import pytest

import explain


# ---------------------------------------------------------------------------------------------------------------------
# Rollout and map statistics (numpy only)
# ---------------------------------------------------------------------------------------------------------------------

def random_attention(rng, heads, t):
    a = rng.random((heads, t, t))
    return a / a.sum(-1, keepdims=True)


def test_rollout_hand_computed_order_and_row():
    # R1 = [[1,0,0],[.5,.5,0],[0,0,1]], R2 = [[.5,.5,0],[0,1,0],[0,.5,.5]] do not commute:
    # row 0 of R2 @ R1 = [.75, .25, 0]; row 0 of R1 @ R2 would give [.5, 0]; column 0 of R2 @ R1 would give [.5, .25].
    a1 = np.array([[[1., 0, 0], [1, 0, 0], [0, 0, 1]]])
    a2 = np.array([[[0., 1, 0], [0, 1, 0], [0, 1, 0]]])
    np.testing.assert_allclose(explain.attention_rollout([a1, a2]), [0.25, 0.0])
    np.testing.assert_allclose(explain.attention_rollout([a2, a1]), [0.5, 0.0])


def test_rollout_shape_and_cls_row_sums_to_one():
    rng = np.random.default_rng(0)
    r = explain.attention_rollout([random_attention(rng, 3, 17) for _ in range(4)])
    assert r.shape == (16,) and (r >= 0).all() and 0 < r.sum() < 1


def test_rollout_identity_and_uniform_cases():
    eye = np.eye(5)[None].repeat(2, axis=0)
    np.testing.assert_allclose(explain.attention_rollout([eye, eye, eye]), 0.0)
    uniform = np.full((2, 5, 5), 1 / 5)
    np.testing.assert_allclose(explain.attention_rollout([uniform]), [0.1] * 4)   # CLS row [.6, .1, .1, .1, .1]


def test_rollout_batch_dim_fusions_and_discard_keeps_cls_to_cls():
    rng = np.random.default_rng(1)
    layers = [random_attention(rng, 2, 10) for _ in range(2)]
    np.testing.assert_allclose(explain.attention_rollout(layers), explain.attention_rollout([a[None] for a in layers]))
    for fusion in ("max", "min"):
        assert explain.attention_rollout(layers, head_fusion=fusion).shape == (9,)
    fused = np.array([[0.01, 0.99], [0.6, 0.4]])                 # CLS->CLS is the smallest entry
    kept = explain.discard_lowest(fused, 0.75)
    assert kept[0, 0] == 0.01 and kept[1, 1] == 0 and kept[1, 0] == 0 and kept[0, 1] == 0.99


def test_last_layer_cls_attention_and_grid():
    rng = np.random.default_rng(2)
    layers = [random_attention(rng, 3, 17) for _ in range(2)]
    np.testing.assert_allclose(explain.last_layer_cls_attention(layers), layers[-1][:, 0, 1:])
    assert explain.to_grid(np.arange(16)).shape == (4, 4)
    with pytest.raises(ValueError):
        explain.to_grid(np.arange(15))


def test_mass_statistics():
    g = np.zeros((16, 16))
    g[4:12, 4:12] = 1
    assert explain.centre_mass(g) == 1.0 and explain.border_mass(g) == 0.0
    assert explain.centre_mass(np.ones((16, 16))) == 0.25
    assert explain.border_mass(np.ones((16, 16))) == pytest.approx(60 / 256)
    assert np.isnan(explain.centre_mass(np.zeros((4, 4))))
    assert explain.top_mass([0, 0, 5, 5], k=2) == 1.0
    assert explain.top_overlap([9, 0, 8, 7], {0, 3}, k=2) == 0.5
    blob = explain.centre_blob()
    assert blob.shape == (16, 16) and np.unravel_index(blob.argmax(), blob.shape) in {(7, 7), (7, 8), (8, 7), (8, 8)}
    assert explain.centre_mass(blob) == pytest.approx(0.799, abs=1e-3)


def test_map_statistics_drop_centre_for_uncertain_crops():
    maps = {"rollout": np.ones(16), "gradcam_correct": np.arange(16.0)}
    certain = explain.map_statistics(maps, False, {0})
    uncertain = explain.map_statistics(maps, True, {0})
    assert certain["rollout_centre"] == 0.25 and np.isnan(uncertain["rollout_centre"])
    assert np.isnan(uncertain["gradcam_correct_border"]) and uncertain["rollout_top3"] == certain["rollout_top3"]
    assert "rollout_top3_in_highnorm" in certain and "gradcam_correct_top3_in_highnorm" not in certain


def test_high_norm_tokens_picks_the_most_extreme_layer():
    layer, top, ratio = explain.high_norm_tokens([[1, 1, 2, 1], [1, 10, 1, 1]], k=1)
    assert (layer, top, ratio) == (1, {1}, 10.0)


# ---------------------------------------------------------------------------------------------------------------------
# Categories, sampling, deployments
# ---------------------------------------------------------------------------------------------------------------------

def synthetic_predictions(n=200, seed=0):
    rng = np.random.default_rng(seed)
    p_correct = rng.random(n)
    return pd.DataFrame({"label_uid": [f"city{i % 3}:{i}" for i in range(n)],
                         "truth": rng.choice(["correct", "incorrect"], n),
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
    assert out.pred.tolist() == ["correct", "incorrect"] and out.category.tolist() == ["FP", "FN"]


def test_sample_categories_includes_tn_caps_orders_and_is_deterministic():
    preds = explain.add_verdicts(synthetic_predictions())
    s1 = explain.sample_categories(preds, 10, seed=7, label_type="Obstacle")
    s2 = explain.sample_categories(preds.sample(frac=1, random_state=3), 10, seed=7, label_type="Obstacle")
    assert s1.label_uid.tolist() == s2.label_uid.tolist()
    assert set(s1.category) == {"TP", "FP", "FN", "TN"} and (s1.groupby("category").size() == 10).all()
    for _, g in s1.groupby("category"):
        assert g.confidence.is_monotonic_decreasing and g.tile.tolist() == list(range(1, 11))
    assert s1.overlay_id.iloc[0] == "obstacle-TP-01" and s1.label_uid.is_unique


def test_sample_categories_keeps_everything_in_a_small_category():
    preds = explain.add_verdicts(synthetic_predictions())
    fn = int((preds.category == "FN").sum())
    assert (explain.sample_categories(preds, fn + 5, seed=1).category == "FN").sum() == fn


def test_load_crops_dir_keeps_the_file_stem_not_a_label_uid(tmp_path):
    for cls, name in (("correct", "east_springfield_12.webp"), ("incorrect", "spr_12.webp")):
        d = tmp_path / "test" / cls
        d.mkdir(parents=True)
        (d / name).write_bytes(b"x")
    m = explain.load_crops_dir(str(tmp_path))
    assert m.prefix.tolist() == ["east_springfield", "spr"] and m.label_id.tolist() == ["12", "12"]
    assert "label_uid" not in m.columns
    with pytest.raises(FileNotFoundError):
        explain.load_crops_dir(str(tmp_path), split="val")


def listing(*entries):
    return {"cities": [{"city_id": c, "visibility": v, "url": f"https://sidewalk-{s}.cs.washington.edu"}
                       for c, v, s in entries]}


def test_public_deployments_matches_exact_subdomains_only():
    cj = listing(("springfield-xx", "public", "springfield"), ("east-springfield-xx", "public", "east-springfield"),
                 ("spr-yy", "private", "spr"), ("shelbyville-xx", "private", "shelbyville"),
                 ("ogdenville-xx", "public", "ogden"))
    got = explain.public_deployments(["springfield", "east_springfield", "spr", "spring", "shelbyville", "ogden",
                                      "ogdenville", "nowhere"], cj)
    assert got == {"springfield": "springfield-xx", "east_springfield": "east-springfield-xx",
                   "ogden": "ogdenville-xx"}


def test_public_deployments_alias_only_when_public_and_blocked_by_a_private_namesake():
    pub = listing(("seattle-wa", "public", "seattle"))
    assert explain.public_deployments(["sea"], pub) == {"sea": "seattle-wa"}
    assert explain.public_deployments(["sea"], listing(("seattle-wa", "private", "seattle"))) == {}
    clash = listing(("seattle-wa", "public", "seattle"), ("sea-zz", "private", "sea"))
    assert explain.public_deployments(["sea"], clash) == {}


def test_label_point_uncertain_flags_non_square_crops():
    assert explain.label_point_uncertain([(640, 640), (640, 323), (600, 640)]) == [False, True, True]


# ---------------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------------

def test_strip_contact_sheet_and_packet_panels(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(3)
    image = (rng.random((56, 56, 3)) * 255).astype(np.uint8)
    grids = [rng.random((4, 4)), np.zeros((4, 4))]
    s = explain.strip(image, grids)
    assert s.shape == (56, 168, 3) and s.dtype == np.uint8
    assert not np.array_equal(explain.strip(image, grids, uncertain=True)[:, :56], s[:, :56])
    sheet = tmp_path / "sheet.jpg"
    explain.contact_sheet([s] * 5, [f"{i:02d}" for i in range(5)], str(sheet), ncols=2, tile_height=40, header="h")
    with Image.open(sheet) as im:
        assert im.format == "JPEG" and im.height > 3 * 40
    im = explain.labelled_panels(image, grids, ["panel A", "panel B"], title="t")
    assert im.width == 3 * (56 + 6) + 6


def test_denormalise_round_trip():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image = np.random.default_rng(4).integers(0, 256, (8, 8, 3)).astype(np.uint8)
    x = ((image / 255.0) - mean) / std
    np.testing.assert_array_equal(explain.denormalise(x.transpose(2, 0, 1), mean, std), image)


# ---------------------------------------------------------------------------------------------------------------------
# Packet, unblinding, kappa, go/no-go
# ---------------------------------------------------------------------------------------------------------------------

def fake_samples(n=8):
    return pd.DataFrame({"overlay_id": [f"x-{c}-{i:02d}" for i, c in zip(range(n), ["TP", "FP", "FN", "TN"] * n)],
                         "label_type": "X", "category": (["TP", "FP", "FN", "TN"] * n)[:n],
                         "label_uid": [f"springfield-xx:{i}" for i in range(n)],
                         "label_point_uncertain": [False] * (n - 1) + [True]})


def test_packet_key_is_blinded_deterministic_and_complete():
    s = fake_samples()
    k1, k2 = explain.packet_key(s, seed=1), explain.packet_key(s.iloc[::-1], seed=1)
    pd.testing.assert_frame_equal(k1, k2)
    assert list(k1.columns) == explain.KEY_COLUMNS and k1.packet_id.is_unique
    assert sorted(k1.overlay_id) == sorted(s.overlay_id)
    assert k1.overlay_id.tolist() != sorted(s.overlay_id)                      # shuffled
    for row in k1.itertuples():
        assert sorted(getattr(row, f"panel_{p}") for p in explain.PANEL_LETTERS) == sorted(explain.PACKET_METHODS)
    sheet = explain.blank_sheet(k1, dict(zip(s.overlay_id, s.label_point_uncertain)))
    assert list(sheet.columns) == explain.SHEET_COLUMNS
    assert not set(sheet.columns) & {"category", "truth", "pred", "confidence", "overlay_id", "label_uid"}
    assert sheet.label_point_uncertain.sum() == 1


def test_unblind_maps_panels_back_to_methods():
    s = fake_samples(4)
    key = explain.packet_key(s, seed=2)
    sheet = explain.blank_sheet(key, dict(zip(s.overlay_id, s.label_point_uncertain)))
    for p in explain.PANEL_LETTERS:                      # code "yes" exactly where the panel is the centre blob
        sheet[f"covers_{p}"] = np.where(key[f"panel_{p}"] == "centre_blob", "yes", "no")
    sheet["best_panel"] = "A"
    long, best = explain.unblind(sheet, key)
    assert set(long[long.covers == "yes"].method) == {"centre_blob"}
    assert len(long) == 4 * len(explain.PACKET_METHODS)
    assert best.best_method.tolist() == key.panel_A.tolist()
    with pytest.raises(ValueError):
        explain.unblind(sheet.assign(packet_id="pzzzzzz"), key)


def test_validate_codes():
    sheet = explain.blank_sheet(explain.packet_key(fake_samples(2), 0), {"x-TP-00": False, "x-FP-01": False})
    explain.validate_codes(sheet)
    sheet.loc[0, "covers_C"] = "maybe"
    with pytest.raises(ValueError, match="covers_C"):
        explain.validate_codes(sheet)


def long_frame(codes, method="gradcam", types=None, categories=None):
    n = len(codes)
    return pd.DataFrame({"overlay_id": [f"o{i}" for i in range(n)], "method": method,
                         "label_type": types or ["X"] * n, "category": categories or ["TP"] * n, "covers": codes})


def test_kappa_hand_computed_and_weighted():
    t = explain.agreement(long_frame(["yes", "yes", "no", "no"]), long_frame(["yes", "yes", "no", "yes"]))
    assert t.n.iloc[0] == 4 and t.percent_agreement.iloc[0] == 0.75 and t.kappa.iloc[0] == pytest.approx(0.5)
    r = explain.kappa_row(["no", "partly", "yes", "yes"], ["partly", "partly", "yes", "no"], explain.COVERS_ORDER)
    assert r["weighted_kappa"] == pytest.approx(1 / 7)          # linear weights; quadratic would give 0.0


def test_kappa_is_nan_when_degenerate_and_skips_blank_unclear():
    a = long_frame(["yes", "yes", "unclear", ""], types=["A", "A", "A", "B"])
    b = long_frame(["yes", "yes", "yes", "yes"], types=["A", "A", "A", "B"])
    t = explain.agreement(a, b, by=("label_type",)).set_index("label_type")
    assert t.loc["A", "n"] == 2 and t.loc["A", "percent_agreement"] == 1.0 and np.isnan(t.loc["A", "kappa"])
    assert "B" not in t.index


def test_agreement_rejects_duplicate_rows():
    a = long_frame(["yes", "no"])
    with pytest.raises(pd.errors.MergeError):
        explain.agreement(pd.concat([a, a]), a)


def test_go_no_go_needs_rate_margin_kappa_and_items():
    n = 30
    cats = ["TP"] * 20 + ["FP"] * 10
    good = ["yes"] * 24 + ["no"] * 6                          # rate 0.8
    good2 = ["yes"] * 23 + ["no"] * 7
    blob = ["yes"] * 15 + ["no"] * 15                         # rate 0.5
    def rater(gc, bl, gx):
        return pd.concat([long_frame(gc, "gradcam", categories=cats), long_frame(bl, "centre_blob", categories=cats),
                          long_frame(gx, "gradxact", categories=cats)], ignore_index=True)
    weak = ["yes"] * 18 + ["no"] * 12                         # rate 0.6
    out = explain.go_no_go(rater(good, blob, weak), rater(good2, blob, weak)).set_index("method")
    assert out.loc["gradcam", "decision"] == "go" and out.loc["gradcam", "n_min"] == n
    assert out.loc["gradxact", "decision"] == "no-go"
    high_blob = ["yes"] * 22 + ["no"] * 8                      # blob rate 0.73: gradcam's margin < 0.2
    out = explain.go_no_go(rater(good, high_blob, weak), rater(good2, high_blob, weak)).set_index("method")
    assert out.loc["gradcam", "decision"] == "no-go"


def test_summarise_counts_certain_rows_and_skips_nan():
    s = pd.DataFrame({"label_type": ["A", "A", "A"], "category": ["TP", "TP", "TP"],
                      "label_point_uncertain": [False, False, True], "rollout_centre": [0.2, 0.4, np.nan],
                      "rollout_top3": [0.1, 0.2, 0.3]})
    t = explain.summarise(s).iloc[0]
    assert t.n == 3 and t.n_certain == 2
    assert t.rollout_centre == pytest.approx(0.3) and t.rollout_top3 == pytest.approx(0.2)


# ---------------------------------------------------------------------------------------------------------------------
# Tiny DINOv2 classifier (torch + transformers)
# ---------------------------------------------------------------------------------------------------------------------

def tiny_config(transformers, size=56):
    return transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=2,
                                     image_size=size, patch_size=14, num_labels=2,
                                     id2label={0: "correct", 1: "incorrect"}, label2id={"correct": 0, "incorrect": 1})


@pytest.fixture(scope="module")
def tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    return transformers.Dinov2ForImageClassification._from_config(tiny_config(transformers),
                                                                 attn_implementation="eager").eval()


def test_explain_one_shapes_signs_and_no_leftover_hooks(tiny_model):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 3, 56, 56)
    r = explain.explain_one(tiny_model, x)
    assert r["rollout"].shape == (16,) and r["cls_attention"].shape == (2, 16)
    for maps in (r["gradcam"], r["gradxact"]):
        assert set(maps) == {0, 1} and all(m.shape == (16,) and (m >= 0).all() for m in maps.values())
    assert r["layer_norms"].shape == (2, 16)
    with torch.no_grad():
        probs = tiny_model(pixel_values=x).logits.softmax(-1)[0].numpy()
    np.testing.assert_allclose(r["probs"], probs, rtol=1e-5)
    assert r["target"] == int(probs.argmax())
    assert all(p.grad is None for p in tiny_model.parameters())
    assert explain.gradcam_module(tiny_model) is tiny_model.dinov2.encoder.layer[-1].norm1
    assert explain.block_module(tiny_model) is tiny_model.dinov2.encoder.layer[-1]
    for layer in tiny_model.dinov2.encoder.layer:
        assert not layer._forward_hooks and not layer.norm1._forward_hooks


def test_gradient_maps_match_an_autograd_reference(tiny_model):
    torch = pytest.importorskip("torch")
    x = torch.randn(1, 3, 56, 56)
    r = explain.explain_one(tiny_model, x, target_class=1)
    acts = {}
    h1 = tiny_model.dinov2.encoder.layer[1].norm1.register_forward_hook(lambda m, i, o: acts.__setitem__("n", o))
    h2 = tiny_model.dinov2.encoder.layer[1].register_forward_hook(lambda m, i, o: acts.__setitem__("b", o[0]))
    z = tiny_model(pixel_values=x).logits[0]
    h1.remove()
    h2.remove()
    gn, gb = torch.autograd.grad(z[1] - z[0], [acts["n"], acts["b"]])
    raw = (acts["n"][0, 1:] @ gn[0, 1:].mean(0)).detach().numpy()           # pre-ReLU, patch tokens 1..16
    np.testing.assert_allclose(r["gradcam"][1], np.maximum(raw, 0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(r["gradcam"][0], np.maximum(-raw, 0), rtol=1e-4, atol=1e-6)   # 2 classes: negated
    gxa = (acts["b"][0, 1:] * gb[0, 1:]).sum(-1).detach().numpy()
    np.testing.assert_allclose(r["gradxact"][1], np.maximum(gxa, 0), rtol=1e-4, atol=1e-6)
    assert r["target"] == 1 and np.array_equal(r["gradcam_target"], r["gradcam"][1])


def test_tie_goes_to_the_first_class(tiny_model):
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        w, b = tiny_model.classifier.weight.clone(), tiny_model.classifier.bias.clone()
        tiny_model.classifier.weight.zero_()
        tiny_model.classifier.bias.zero_()
    try:
        assert explain.explain_one(tiny_model, torch.randn(1, 3, 56, 56))["target"] == 0
    finally:
        with torch.no_grad():
            tiny_model.classifier.weight.copy_(w)
            tiny_model.classifier.bias.copy_(b)


def test_grad_x_act_finds_the_only_token_the_head_reads():
    """A head that reads only the last patch token: grad x act must peak there; dropping the wrong token fails."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    base = transformers.Dinov2ForImageClassification._from_config(tiny_config(transformers),
                                                                 attn_implementation="eager").eval()

    class LastToken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dinov2, self.config, self.lin = base.dinov2, base.config, torch.nn.Linear(32, 2)

        def forward(self, pixel_values, output_attentions=False):
            o = self.dinov2(pixel_values=pixel_values, output_attentions=output_attentions)
            return types.SimpleNamespace(logits=self.lin(o.last_hidden_state[:, -1]), attentions=o.attentions)

    model = LastToken().eval()
    hits = 0
    for seed in range(10):
        torch.manual_seed(seed)
        hits += int(np.argmax(explain.explain_one(model, torch.randn(1, 3, 56, 56))["gradxact_target"]) == 15)
    assert hits >= 9


def test_explain_one_under_sdpa_falls_back_to_eager():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.Dinov2ForImageClassification._from_config(tiny_config(transformers, 28),
                                                                  attn_implementation="sdpa").eval()
    assert explain.explain_one(model, torch.randn(1, 3, 28, 28))["rollout"].shape == (4,)
