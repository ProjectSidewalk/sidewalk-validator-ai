"""context_model.py (issue #14): side features, normaliser, heatmap and point mapping (always run), and the three
context arms of the DINOv2 classifier (torch + transformers; skipped on the CPU-only CI). Tiny Dinov2Config backbones
are built from a config and saved locally, never downloaded."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest

import context_model as cm


def test_doctests():
    r = doctest.testmod(cm)
    assert r.attempted > 0 and r.failed == 0


def side_df():
    return pd.DataFrame({
        "label_uid": ["c:1", "c:2", "c:3", "c:4"],
        "severity": [3, None, 1, "2"],
        "tags": ['["a","b"]', "a;b", None, "[]"],
        "image_capture_date": ["2020-09", "2024-09-15", "bad", None],
        "depth_m": [10.0, None, 0, "5.5"],
        "high_quality_user": [True, False, None, "True"],
        "clipped": [0, 1, None, "1"],
    })


def test_side_feature_frame():
    f = cm.side_feature_frame(side_df())
    assert list(f.columns) == cm.SIDE_COLUMNS
    assert f.severity.isna().tolist() == [False, True, False, False] and f.severity_missing.tolist() == [0, 1, 0, 0]
    assert f.n_tags.tolist() == [2, 2, 0, 0]
    assert f.age_years.iloc[0] == pytest.approx(4.0, abs=1e-3) and f.age_years.iloc[1] == 0
    assert f.age_missing.tolist() == [0, 0, 1, 1]
    assert f.log_depth_m.iloc[0] == pytest.approx(np.log(10)) and f.log_depth_m.iloc[3] == pytest.approx(np.log(5.5))
    assert f.depth_missing.tolist() == [0, 1, 1, 0]   # None and 0 are both "no depth"
    assert f.high_quality_user.tolist() == [1, 0, 0, 1] and f.clipped.tolist() == [0, 1, 0, 1]
    # an explicit reference date is used as given (the config's, at inference)
    assert cm.side_feature_frame(side_df(), "2025-09").age_years.iloc[1] == pytest.approx(1.0, abs=1e-2)


def test_normaliser_floors_fills_and_round_trips_through_json():
    f = cm.side_feature_frame(side_df())
    stats = cm.fit_normaliser(f)
    i_clip = cm.SIDE_COLUMNS.index("clipped")
    assert stats["std"][i_clip] > 0
    const = f.assign(high_quality_user=1.0)
    s2 = cm.fit_normaliser(const)
    assert s2["std"][cm.SIDE_COLUMNS.index("high_quality_user")] == 1e-6   # zero std floored
    x = cm.normalise(f, stats)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    i_sev = cm.SIDE_COLUMNS.index("severity")
    med = stats["median"][i_sev]
    assert x[1, i_sev] == pytest.approx((med - stats["mean"][i_sev]) / stats["std"][i_sev])  # NaN -> median
    back = json.loads(json.dumps(stats))
    assert np.array_equal(cm.normalise(f, back), x)
    with pytest.raises(ValueError, match="columns"):
        cm.normalise(f, {**stats, "columns": stats["columns"][::-1]})


def test_all_missing_column_gets_median_zero():
    f = cm.side_feature_frame(side_df().assign(severity=None))
    stats = cm.fit_normaliser(f)
    assert stats["median"][cm.SIDE_COLUMNS.index("severity")] == 0.0
    assert np.isfinite(cm.normalise(f, stats)).all()


def test_heatmap_peak_symmetry_and_sigma():
    m = cm.heatmap(31, 41, 20, 15, 3.0)
    assert m.shape == (31, 41) and m.max() == 1.0 and np.unravel_index(m.argmax(), m.shape) == (15, 20)
    assert m[15, 23] == pytest.approx(np.exp(-0.5)) and m[18, 20] == pytest.approx(np.exp(-0.5))
    assert np.allclose(m[:, 20 - 5], m[:, 20 + 5])
    a = cm.heatmap(31, 41, 20, 15, (6.0, 3.0))
    assert a[15, 26] == pytest.approx(np.exp(-0.5)) and a[18, 20] == pytest.approx(np.exp(-0.5))


def test_heatmap_sigma_scale_and_floor():
    assert cm.heatmap_sigma(1024, 224) == 56.0          # a whole-view crop: a quarter of the input
    assert cm.heatmap_sigma(0, 224) == cm.HEATMAP_SIGMA_FLOOR
    assert cm.heatmap_sigma(400, 448) == 2 * cm.heatmap_sigma(400, 224)
    assert cm.heatmap_sigma(30, 224, floor=1.0) == pytest.approx(30 * 224 / 2048 / 2)


def test_map_point_identity_and_flip():
    assert cm.map_point(5, 7, (0, 0, 20, 20), 20) == (5.0, 7.0)
    assert cm.map_point(5, 7, (0, 0, 20, 20), 20, flip=True) == (14.0, 7.0)


def test_frame_columns_cover_what_the_arms_read():
    assert set(cm.FRAME_COLUMNS) == set(cm.ARMS)
    for cols in cm.FRAME_COLUMNS.values():
        assert {"label_uid", "split"} <= set(cols)
    assert {"severity", "tags", "image_capture_date", "depth_m", "high_quality_user", "clipped"} <= set(
        cm.FRAME_COLUMNS["side_info"])


def test_point_mapping_matches_torchvision_resized_crop():
    """A small blob moved by F.resized_crop (and a flip), as the training transform calls it, lands where map_point
    says (its intensity centroid; a blob rather than one pixel, which a downsampling resize can step over)."""
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchvision.transforms.functional")
    s, res = 448, 224
    for (cy, cx), crop, flip in [((223, 223), (40, 60, 360, 300), False), ((223, 223), (0, 0, 448, 448), True),
                                 ((100, 300), (50, 180, 220, 260), True)]:
        img = torch.from_numpy(cm.heatmap(s, s, cx, cy, 4.0))[None]
        out = F.resized_crop(img, *crop, [res, res], antialias=True)
        if flip:
            out = F.hflip(out)
        w = out[0].numpy()
        yy, xx = np.mgrid[0:res, 0:res]
        got = (float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum()))
        want = cm.map_point(cx, cy, crop, res, flip)
        assert got == pytest.approx(want, abs=0.6), (crop, flip)


# ---------------------------------------------------------------------------------------------------------------------
# torch + transformers: tiny backbones built from a config, saved locally (no network)
# ---------------------------------------------------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

ID2LABEL = {0: "correct", 1: "incorrect"}
LABEL2ID = {"correct": 0, "incorrect": 1}
CROP = 28  # the tiny processor's centre crop (2 x 2 patches of 14 px)


def tiny_base(tmp_path):
    """A 2-layer, 32-wide DINOv2 and a matching slow image processor in ``tmp_path/base``."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch
    torch.manual_seed(0)
    base = tmp_path / "base"
    cfg = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                                    patch_size=14, image_size=56)
    transformers.Dinov2Model(cfg).save_pretrained(base)
    tiny_processor().save_pretrained(base)
    return str(base)


def tiny_processor():
    from transformers import BitImageProcessor
    return BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": CROP, "width": CROP},
                             do_convert_rgb=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])


def build(base, arm, **fields):
    _, model_cls = cm.build_classes()
    if arm == "side_info":
        fields.setdefault("side_features", list(cm.SIDE_COLUMNS))
    return model_cls.from_base(base, arm, ID2LABEL, LABEL2ID, **fields)


def inputs_for(arm, n=2, res=56, side=None):
    import torch
    g = torch.Generator().manual_seed(1)
    kw = {"pixel_values": torch.randn(n, 4 if arm == "view_heatmap" else 3, res if arm == "view_heatmap" else CROP,
                                      res if arm == "view_heatmap" else CROP, generator=g)}
    if arm == "two_scale":
        kw["pixel_values_wide"] = torch.randn(n, 3, CROP, CROP, generator=g)
    if arm == "side_info":
        kw["side_features"] = torch.randn(n, len(cm.SIDE_COLUMNS), generator=g)
    return kw


@pytest.mark.parametrize("arm", cm.ARMS)
def test_model_round_trip_head_and_loss(tmp_path, arm):
    torch = pytest.importorskip("torch")
    base = tiny_base(tmp_path)
    m = build(base, arm).eval()
    assert m.config.model_type == cm.MODEL_TYPE
    assert m.classifier.in_features == {"two_scale": 128, "view_heatmap": 64, "side_info": 64 + 9}[arm]
    kw = inputs_for(arm)
    out = m(**kw, labels=torch.tensor([0, 1]))
    assert out.logits.shape == (2, 2) and out.loss.ndim == 0 and torch.isfinite(out.loss)
    m.save_pretrained(tmp_path / "ckpt")
    cfg = json.loads((tmp_path / "ckpt" / "config.json").read_text())
    assert cfg["context_arm"] == arm and cfg["num_channels"] == (4 if arm == "view_heatmap" else 3)
    _, model_cls = cm.build_classes()
    m2 = model_cls.from_pretrained(tmp_path / "ckpt").eval()
    with torch.no_grad():
        assert torch.allclose(m(**kw).logits, m2(**kw).logits, atol=1e-6)


def test_four_channel_projection_starts_as_the_pretrained_model(tmp_path):
    torch = pytest.importorskip("torch")
    Dinov2Model = pytest.importorskip("transformers").Dinov2Model
    base = tiny_base(tmp_path)
    m = build(base, "view_heatmap").eval()
    w = m.dinov2.embeddings.patch_embeddings.projection.weight
    ref = Dinov2Model.from_pretrained(base).eval()
    w0 = ref.embeddings.patch_embeddings.projection.weight
    assert w.shape[1] == 4 and torch.equal(w[:, :3], w0) and (w[:, 3] == 0).all()
    x = torch.randn(2, 4, 56, 56)
    with torch.no_grad():
        a = m.dinov2(pixel_values=x).last_hidden_state
        b = ref(pixel_values=x[:, :3]).last_hidden_state
    assert torch.allclose(a, b, atol=1e-5)  # the heatmap channel changes nothing until it is trained


def test_arm_inputs_are_required(tmp_path):
    base = tiny_base(tmp_path)
    for arm, missing in (("two_scale", "pixel_values_wide"), ("side_info", "side_features")):
        kw = inputs_for(arm)
        kw.pop(missing)
        with pytest.raises(ValueError, match=missing):
            build(base, arm)(**kw)


def write_context_dataset(root, uids, view_px=112):
    """A tiny ``<root>/Lt`` with crops, a context index (wide and view WebPs with structure) and split rows."""
    import context_crops as cc
    from PIL import Image
    base = root / "Lt"
    rng = np.random.default_rng(0)
    rows = []
    with cc.ContextIndex(str(base / "context")) as ctx:
        for k, (uid, split) in enumerate(uids):
            stem = f"{'correct' if k % 2 == 0 else 'incorrect'}/c_{k}"
            for sub, size in (("", 64), ("context/wide/", 64), ("context/view/", view_px)):
                p = base / f"{sub}{stem}.webp"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(p, quality=95)
            ctx.write(uid, {"wide_path": f"wide/{stem}.webp", "view_path": f"view/{stem}.webp", "wide_half_px": 800,
                            "wide_clipped": 0})
            rows.append({"label_uid": uid, "city": "c", "label_id": k, "pano_id": f"p{k // 2}", "label_type": "Lt",
                         "label_class": "correct" if k % 2 == 0 else "incorrect", "crop_path": f"{stem}.webp",
                         "split": split, "hard": 0, "crop_half_px": 200 + 10 * k, "severity": 1 + k % 3,
                         "tags": '["a"]' if k % 2 else "[]", "image_capture_date": f"20{18 + k % 6}-05",
                         "depth_m": 6.0 + k, "high_quality_user": bool(k % 2), "clipped": 0})
    df = pd.DataFrame(rows)
    df.to_csv(base / "split.csv", index=False)
    return df


def frames(df, arm, root):
    out = []
    for s in ("train", "val"):
        part = df[df.split == s].copy()
        part["image"] = [str(root / "Lt" / p) for p in part.crop_path]
        part["label"] = part.label_class.map(LABEL2ID)
        out.append(part[["image", "label", *cm.FRAME_COLUMNS[arm]]].reset_index(drop=True))
    return out


def base_collate(examples):
    import torch
    return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e["label"] for e in examples])}


def examples_of(frame, n=2):
    import torch
    return [{**r, "pixel_values": torch.zeros(3, CROP, CROP)} for r in frame.head(n).to_dict("records")]


UIDS = [(f"c:{k}", "train" if k < 4 else "val") for k in range(6)]


@pytest.mark.parametrize("arm", cm.ARMS)
def test_build_for_training_collate(tmp_path, arm):
    torch = pytest.importorskip("torch")
    base = tiny_base(tmp_path)
    df = write_context_dataset(tmp_path, UIDS)
    train, val = frames(df, arm, tmp_path)
    args = SimpleNamespace(context_arm=arm, context_view_res=56, data_dir=str(tmp_path), label_type="Lt")
    model, collate = cm.build_for_training(args, base, train, val, ID2LABEL, LABEL2ID, base_collate, tiny_processor())
    tr, va = examples_of(train), examples_of(val)
    torch.manual_seed(0)
    b1, b2 = collate(tr), collate(tr)
    v1, v2 = collate(va), collate(va)
    if arm == "two_scale":
        assert b1["pixel_values_wide"].shape == (2, 3, CROP, CROP)
        assert not torch.equal(b1["pixel_values_wide"], b2["pixel_values_wide"])   # augmented
        assert torch.equal(v1["pixel_values_wide"], v2["pixel_values_wide"])       # deterministic
    elif arm == "view_heatmap":
        assert b1["pixel_values"].shape == (2, 4, 56, 56)
        assert not torch.equal(b1["pixel_values"], b2["pixel_values"])
        assert torch.equal(v1["pixel_values"], v2["pixel_values"])
        hm = v1["pixel_values"][0, 3].numpy()
        assert hm.max() > 0.9 and abs(np.unravel_index(hm.argmax(), hm.shape)[0] - 27.5) <= 0.5  # centred
        assert model.config.num_channels == 4 and model.config.index_sha256
    else:
        assert b1["side_features"].shape == (2, len(cm.SIDE_COLUMNS))
        assert torch.equal(v1["side_features"], v2["side_features"])
        cfg = model.config
        assert cfg.side_stats["columns"] == cm.SIDE_COLUMNS and cfg.age_reference_date == "2021-05"
    with torch.no_grad():
        out = model(**{k: v for k, v in b1.items()})
    assert out.logits.shape == (2, 2) and torch.isfinite(out.loss)


def test_missing_context_row_fails_before_the_model_is_built(tmp_path):
    df = write_context_dataset(tmp_path, UIDS)
    train, val = frames(df, "two_scale", tmp_path)
    val.loc[0, "label_uid"] = "c:404"
    args = SimpleNamespace(context_arm="two_scale", context_view_res=224, data_dir=str(tmp_path), label_type="Lt")
    with pytest.raises(ValueError, match="c:404"):
        cm.build_for_training(args, str(tmp_path / "no-such-base"), train, val, ID2LABEL, LABEL2ID, base_collate,
                              SimpleNamespace(crop_size={"height": 224, "width": 224}, image_mean=[0] * 3,
                                              image_std=[1] * 3))
    with pytest.raises(SystemExit, match="split.csv"):
        cm.build_for_training(args, "x", train.drop(columns="split"), val, ID2LABEL, LABEL2ID, base_collate, None)


@pytest.mark.parametrize("arm", cm.ARMS)
def test_predict_writes_predictions_evaluate_reads(tmp_path, arm):
    import evaluate
    base = tiny_base(tmp_path)
    uids = [(f"c:{k}", ["train", "val", "test", "test_heldout"][(k // 2) % 4]) for k in range(16)]  # both classes
    df = write_context_dataset(tmp_path, uids)
    train, val = frames(df, arm, tmp_path)
    args = SimpleNamespace(context_arm=arm, context_view_res=224, data_dir=str(tmp_path), label_type="Lt")
    model, _ = cm.build_for_training(args, base, train, val, ID2LABEL, LABEL2ID, base_collate, tiny_processor())
    ckpt = tmp_path / "run" / "checkpoint-1"
    model.save_pretrained(ckpt)
    tiny_processor().save_pretrained(ckpt)
    assert cm.main(["predict", "--label_type", "Lt", "--data-dir", str(tmp_path), "--checkpoint", str(ckpt),
                    "--tag", arm]) == 0
    pred = pd.read_csv(tmp_path / "Lt" / f"predictions_{arm}.csv")
    assert list(pred.columns) == evaluate.PREDICTION_COLUMNS and len(pred) == 12
    assert np.allclose(pred.p_correct + pred.p_incorrect, 1, atol=1e-5)
    m = evaluate.report(pred)
    assert m["test"]["n"] == 4 and m["test_heldout"]["n"] == 4 and m["test"]["auc"] is not None
