"""export_onnx.py: pure parity/served-vote/table helpers, and a tiny random DINOv2 round trip through ONNX + int8.

The round trip builds a 2-layer Dinov2ForImageClassification on 28 x 28 inputs in a temp dir, so nothing is
downloaded; it is skipped where torch, transformers, onnx or onnxruntime are missing (CI's CPU-only install).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from export_onnx import list_crops, markdown_table, parity, served_votes, softmax, table


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------------

def test_doctests():
    import doctest

    import export_onnx
    assert doctest.testmod(export_onnx).failed == 0

def test_softmax_rows_sum_to_one_and_are_stable():
    p = softmax(np.array([[1000.0, 1000.0], [0.0, np.log(3.0)]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[0], [0.5, 0.5])
    assert np.allclose(p[1], [0.25, 0.75])


def test_parity_identical_logits_is_zero():
    x = np.array([[2.0, -1.0], [0.3, 0.1], [-4.0, 4.0]])
    r = parity(x, x.copy())
    assert (r["n"], r["max_abs_dlogit"], r["max_abs_dprob"], r["verdict_flips"], r["vote_changes"]) == (3, 0, 0, 0, 0)
    assert r["votes"] == r["votes_ref"] == 2


def test_parity_counts_flips_and_raw_gate_crossings():
    # row 0: confidence 0.952 -> 0.881, crosses the raw 0.92 line, same verdict
    # row 1: verdict flips (correct 0.55 -> incorrect 0.55), neither side votes: a flip, not a vote change
    # row 2: unchanged, votes on both sides
    ref = np.array([[3.0, 0.0], [0.2, 0.0], [5.0, 0.0]])
    var = np.array([[2.0, 0.0], [0.0, 0.2], [5.0, 0.0]])
    r = parity(ref, var)
    assert r["verdict_flips"] == 1
    assert r["votes_ref"] == 2 and r["votes"] == 1
    assert r["vote_changes"] == 1
    assert r["max_abs_dlogit"] == pytest.approx(1.0)
    assert r["max_abs_dprob"] == pytest.approx(np.abs(softmax(ref) - softmax(var)).max())   # row 1: 0.55 -> 0.45


def test_parity_verdict_flip_between_two_votes_is_a_vote_change():
    r = parity(np.array([[5.0, 0.0]]), np.array([[0.0, 5.0]]))
    assert r["votes_ref"] == 1 and r["votes"] == 1 and r["vote_changes"] == 1


def test_parity_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        parity(np.zeros((2, 2)), np.zeros((3, 2)))


def test_served_votes_uses_the_api_map_lookup():
    # 'correct' map: 0.0 below 0.9, 0.95 from 0.9 up; 'incorrect' never reaches the gate (a held-back class)
    m = {"correct": {"0.5": 0.0, "0.9": 0.95, "1.0": 0.97}, "incorrect": {"0.5": 0.0, "1.0": 0.9199}}
    verdict, votes = served_votes([0.95, 0.85, 0.01], [0.05, 0.15, 0.99], m)
    assert verdict.tolist() == ["correct", "correct", "incorrect"]
    assert votes.tolist() == [True, False, False]
    # without a map the raw-confidence proxy would have let the Disagree verdict vote
    assert served_votes([0.01], [0.99])[1].tolist() == [True]


def test_served_votes_tie_goes_to_correct_like_the_api():
    verdict, _ = served_votes([0.5], [0.5])
    assert verdict.tolist() == ["correct"]


def test_list_crops_reads_the_hf_layout(tmp_path):
    for cls, name in [("correct", "seattle_12.webp"), ("incorrect", "new_york_7.webp"), ("correct", "a_b_3.png"),
                      ("correct", "notes.txt")]:
        d = tmp_path / "test" / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"x")
    df = list_crops(str(tmp_path))
    assert df.label_uid.tolist() == ["a_b:3", "new_york:7", "seattle:12"]   # city is up to the LAST underscore
    assert df.label_class.tolist() == ["correct", "incorrect", "correct"]


def test_list_crops_without_split_level_and_empty(tmp_path):
    (tmp_path / "incorrect").mkdir()
    (tmp_path / "incorrect" / "x_1.webp").write_bytes(b"x")
    assert list_crops(str(tmp_path)).label_uid.tolist() == ["x:1"]
    with pytest.raises(FileNotFoundError):
        list_crops(str(tmp_path / "missing"))


def test_markdown_table_formats_and_dashes_missing():
    df = pd.DataFrame({"variant": ["a", "b"], "auc": [0.91234, None], "n": [3, 4]})
    out = markdown_table(df)
    assert out.splitlines() == ["| variant | auc | n |", "|---|---|---|", "| a | 0.9123 | 3 |", "| b | — | 4 |"]


def test_table_concatenates_bench_csvs(tmp_path):
    cols = {"label_type": "X", "variant": "v", "size_mb": 1.0, "params_m": 2.0, "accuracy": 0.5, "macro_f1": 0.5,
            "auc": 0.5, "r_at_p95_correct": 0.1, "r_at_p95_incorrect": 0.2, "vs_ref_max_abs_dprob": 0.0,
            "vs_ref_verdict_flips": 0, "vs_ref_vote_changes": 0, "lat_b1_ms": 10.0}
    for i in range(2):
        pd.DataFrame([{**cols, "variant": f"v{i}"}]).to_csv(tmp_path / f"bench_{i}.csv", index=False)
    out = table([str(tmp_path / "bench_0.csv"), str(tmp_path / "bench_1.csv")])
    assert out.count("\n") == 4 and "lat_b1_ms" in out.splitlines()[0]


# ---------------------------------------------------------------------------------------------------------------------
# Round trip on a tiny random model (no downloads)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=28,
                                       patch_size=14, num_labels=2, id2label={0: "correct", 1: "incorrect"},
                                       label2id={"correct": 0, "incorrect": 1})
    model = transformers.Dinov2ForImageClassification(config).eval()
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    transformers.BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 28, "width": 28}).save_pretrained(ckpt)
    return model, str(ckpt), tmp_path


def test_export_quantize_round_trip(tiny_checkpoint):
    import onnxruntime as ort
    import torch

    from export_onnx import export_model, find_beside, id2label_order, onnx_param_count, quantize_model

    model, ckpt, tmp = tiny_checkpoint
    out = str(tmp / "onnx")
    fp32 = export_model(ckpt, out)
    int8 = quantize_model(fp32)

    info = json.load(open(os.path.join(out, "export_info.json")))
    assert info["opset"] == 17 and info["input"]["pixel_values"] == ["batch", 3, 28, 28]
    assert set(info["files"]) == {"model.onnx", "model_int8.onnx"}
    assert info["parameters"] == sum(p.numel() for p in model.parameters())
    # the graph drops the unused mask token and stores the position embeddings already interpolated to 28 x 28
    assert 0.9 * info["parameters"] < onnx_param_count(fp32) <= info["parameters"]
    assert find_beside(int8, "preprocessor_config.json") and id2label_order(find_beside(int8, "config.json")) == [0, 1]

    x = torch.randn(5, 3, 28, 28)   # batch of 5: the batch axis is dynamic
    with torch.no_grad():
        ref = model(pixel_values=x).logits.numpy()
    for path, tol in ((fp32, 1e-4), (int8, 0.5)):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"pixel_values": x.numpy()})[0]
        assert got.shape == (5, 2)
        assert parity(ref, got)["max_abs_dlogit"] < tol, path
    assert os.path.getsize(int8) < os.path.getsize(fp32)


def test_weight_only_int8_keeps_activations_fp32(tiny_checkpoint):
    pytest.importorskip("onnx_ir")   # onnxruntime's MatMulNBits quantizer imports it
    import onnx
    import onnxruntime as ort
    import torch

    from export_onnx import export_model, quantize_model

    model, ckpt, tmp = tiny_checkpoint
    out = str(tmp / "onnx")
    wo = quantize_model(export_model(ckpt, out), method="weight-only", block_size=16)
    assert os.path.basename(wo) == "model_int8_wo.onnx"
    ops = {n.op_type for n in onnx.load(wo).graph.node}
    assert "MatMulNBits" in ops and "MatMulInteger" not in ops and "DynamicQuantizeLinear" not in ops
    info = json.load(open(os.path.join(out, "export_info.json")))
    assert info["files"]["model_int8_wo.onnx"]["quantization"]["bits"] == 8

    x = torch.randn(3, 3, 28, 28)
    with torch.no_grad():
        ref = model(pixel_values=x).logits.numpy()
    got = ort.InferenceSession(wo, providers=["CPUExecutionProvider"]).run(None, {"pixel_values": x.numpy()})[0]
    assert parity(ref, got)["max_abs_dlogit"] < 0.05


def test_quantize_rejects_unknown_method(tmp_path):
    from export_onnx import quantize_model
    with pytest.raises(ValueError):
        quantize_model(str(tmp_path / "model.onnx"), method="int4")


# ---------------------------------------------------------------------------------------------------------------------
# Boundaries and helpers the review's mutants slipped past
# ---------------------------------------------------------------------------------------------------------------------

def test_served_votes_boundaries_match_the_api():
    # A confidence equal to a map key gets that key's value (the API keeps the last threshold <= c), and a served
    # value of exactly 0.92 votes (SidewalkWebpage's >= ai-validation-min-accuracy).
    m = {"correct": {"0.5": 0.0, "0.9": 0.92, "0.95": 0.99}, "incorrect": {"0.5": 0.0, "0.9": 0.9199}}
    _, votes = served_votes([0.9, 0.8999], [0.1, 0.1001], m)
    assert votes.tolist() == [True, False]
    _, votes = served_votes([0.1], [0.9], m)            # incorrect at its key, value 0.9199: below the gate
    assert votes.tolist() == [False]
    _, votes = served_votes([0.92, 0.9199], [0.08, 0.0801])   # raw proxy: >= 0.92 votes
    assert votes.tolist() == [True, False]


def test_softmax_is_float32_like_the_api():
    assert softmax(np.array([[0.0, 1.0]])).dtype == np.float32


def test_accuracy_row_known_values():
    from export_onnx import accuracy_row
    p_inc = np.array([0.1, 0.2, 0.6, 0.4, 0.9])
    logits = np.log(np.stack([1 - p_inc, p_inc], axis=1))
    r = accuracy_row(np.array(["correct", "correct", "correct", "incorrect", "incorrect"]), logits)
    # verdicts c,c,i,c,i: 3 of 5 right; F1 correct 2/3, incorrect 1/2; AUC of p_incorrect 5/6
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert r["auc"] == pytest.approx(5 / 6)


def test_median_ms_warms_up_then_takes_the_median(monkeypatch):
    import export_onnx
    calls = []
    clock = iter([0.0, 0.001, 1.0, 1.005, 2.0, 2.002])   # timed calls of 1, 5 and 2 ms
    monkeypatch.setattr(export_onnx.time, "perf_counter", lambda: next(clock))
    assert export_onnx.median_ms(lambda: calls.append(1), warmup=2, runs=3) == pytest.approx(2.0)
    assert len(calls) == 5


def test_sample_takes_the_first_b_and_tiles():
    from export_onnx import _sample
    x = np.arange(3).reshape(3, 1)
    assert _sample(x, 2).ravel().tolist() == [0, 1]
    assert _sample(x, 5).ravel().tolist() == [0, 1, 2, 0, 1]


def test_public_crops_drops_private_unlisted_and_ambiguous():
    from export_onnx import public_crops
    df = pd.DataFrame({"label_uid": ["seattle:1", "st_louis:2", "secret:3", "nowhere:4", "springfield:5", "sea:6",
                                     "sec:7"]})
    url = "https://sidewalk-{}.cs.washington.edu"
    listing = {"cities": [{"city_id": "seattle-wa", "visibility": "public", "url": url.format("sea")},
                          {"city_id": "st-louis-mo", "visibility": "public"},
                          {"city_id": "secret-xx", "visibility": "private", "url": url.format("sec")},
                          {"city_id": "springfield-ma", "visibility": "public"},
                          {"city_id": "springfield-il", "visibility": "public"}]}
    kept, n = public_crops(df, listing)
    # schema names and public subdomains kept; private (by name or subdomain), unlisted and ambiguous dropped
    assert kept.label_uid.tolist() == ["seattle:1", "st_louis:2", "sea:6"] and n == 4


def test_public_city_rule_rejects_a_private_subdomain_that_collides_with_a_public_name():
    from export_onnx import public_city_rule
    url = "https://sidewalk-{}.cs.washington.edu"
    listing = {"cities": [{"city_id": "alpha-zz", "visibility": "public", "url": url.format("alp")},
                          {"city_id": "beta-zz", "visibility": "private", "url": url.format("alpha")}]}
    rule = public_city_rule(listing)
    # "alpha" matches the public city_id alpha-zz by name, but it is also the private deployment's subdomain: a
    # crop named alpha_<id> could come from either, so it is refused (fail closed)
    assert rule("alp") and not rule("alpha") and not rule("beta")


def test_export_runs_the_checker_and_quantized_graphs_have_the_promised_types(tiny_checkpoint, monkeypatch):
    pytest.importorskip("onnx_ir")
    import onnx
    from onnx import numpy_helper

    import export_onnx

    _, ckpt, tmp = tiny_checkpoint
    checked = []
    real = onnx.checker.check_model
    monkeypatch.setattr(onnx.checker, "check_model", lambda p, *a, **k: (checked.append(p), real(p, *a, **k)))
    fp32 = export_onnx.export_model(ckpt, str(tmp / "onnx"))
    assert checked == [fp32]

    def graph(path):
        m = onnx.load(path)
        return m.graph.node, {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}

    nodes, inits = graph(export_onnx.quantize_model(fp32))
    assert any(n.op_type == "DynamicQuantizeLinear" for n in nodes)
    weights = [a for k, a in inits.items() if k.endswith("_quantized")]
    assert weights and all(a.dtype == np.int8 for a in weights)          # QInt8 weights, not QUInt8
    matmul_scales = [a for k, a in inits.items() if k.endswith("_scale") and "MatMul" in k]
    assert matmul_scales and all(a.size == 1 for a in matmul_scales)   # per tensor by default

    _, inits = graph(export_onnx.quantize_model(fp32, str(tmp / "onnx" / "pc.onnx"), per_channel=True))
    assert any(a.size > 1 for k, a in inits.items() if k.endswith("_scale"))

    nodes, _ = graph(export_onnx.quantize_model(fp32, method="weight-only", block_size=16))
    nbits = [n for n in nodes if n.op_type == "MatMulNBits"]
    assert nbits and all({a.name: a.i for a in n.attribute}["bits"] == 8 for n in nbits)


def test_score_batch_is_forced_to_1_for_dynamic_graphs(tiny_checkpoint):
    import export_onnx
    _, ckpt, tmp = tiny_checkpoint
    fp32 = export_onnx.export_model(ckpt, str(tmp / "onnx"))
    int8 = export_onnx.quantize_model(fp32)
    assert export_onnx.effective_score_batch(fp32, 8) == 8
    with pytest.warns(UserWarning, match="DynamicQuantizeLinear"):
        assert export_onnx.effective_score_batch(int8, 8) == 1
    assert export_onnx.make_session(fp32, 3).get_session_options().intra_op_num_threads == 3


def test_bench_end_to_end_on_a_tiny_model(tiny_checkpoint):
    from PIL import Image

    import export_onnx

    _, ckpt, tmp = tiny_checkpoint
    fp32 = export_onnx.export_model(ckpt, str(tmp / "onnx"))
    int8 = export_onnx.quantize_model(fp32)
    rng = np.random.default_rng(0)
    for cls, names in (("correct", ["seattle_1", "seattle_2", "hidden_3"]), ("incorrect", ["seattle_4", "seattle_5"])):
        d = tmp / "crops" / "test" / cls
        d.mkdir(parents=True)
        for n in names:
            Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(d / f"{n}.png")
    url = "https://sidewalk-{}.cs.washington.edu"
    (tmp / "cities.json").write_text(json.dumps({"cities": [
        {"city_id": "seattle-wa", "visibility": "public", "url": url.format("sea")},
        {"city_id": "hidden-xx", "visibility": "private", "url": url.format("hid")}]}))
    with pytest.warns(UserWarning, match="DynamicQuantizeLinear"):
        df = export_onnx.main(["bench", "--label-type", "X", "--crops", str(tmp / "crops"), "--model", ckpt,
                               "--onnx", f"onnx-fp32={fp32}", "--onnx", f"onnx-int8={int8}", "--threads", "2",
                               "--batch-sizes", "1,2", "--warmup", "1", "--runs", "2", "--score-batch", "4",
                               "--cities-json", str(tmp / "cities.json"), "--out", str(tmp / "bench")])
    assert df.variant.tolist() == ["torch-fp32-cpu", "onnx-fp32", "onnx-int8"]
    assert (df.n_crops == 4).all() and (df.n_crops_dropped_not_public == 1).all()
    ref, onnx_fp32, q = (df.iloc[i] for i in range(3))
    assert ref.vs_ref_max_abs_dlogit == 0 and onnx_fp32.vs_ref_max_abs_dlogit < 1e-4   # the reference is torch
    assert q.vs_ref_max_abs_dlogit > 0   # int8 is compared with torch, not with itself (reference = first variant)
    assert (ref.score_batch, onnx_fp32.score_batch, q.score_batch) == (4, 4, 1)
    assert pd.isna(onnx_fp32.b16_vs_b1_max_abs_dprob) and q.b16_vs_b1_max_abs_dprob >= 0
    assert (df.ms_per_crop_b2 == df.lat_b2_ms / 2).all()
    saved = pd.read_csv(tmp / "bench" / "bench_X.csv")
    assert len(saved) == 3 and not saved.label_type.isna().any()
    scores = pd.read_csv(tmp / "bench" / "scores_X.csv")
    assert scores.label_uid.tolist() == ["seattle:1", "seattle:2", "seattle:4", "seattle:5"]


def test_dynamic_graphs_are_scored_one_crop_per_call(tiny_checkpoint):
    """bench must score a DynamicQuantizeLinear graph at batch 1 even when asked for more, and measure the gap."""
    import onnxruntime as ort
    from PIL import Image
    from transformers import AutoImageProcessor

    import export_onnx

    _, ckpt, tmp = tiny_checkpoint
    int8 = export_onnx.quantize_model(export_onnx.export_model(ckpt, str(tmp / "onnx")))
    # Crops with very different intensity ranges, so a batch-wide activation scale differs from a per-crop one.
    rng = np.random.default_rng(1)
    for i, (cls, lo, hi) in enumerate([("correct", 0, 20), ("correct", 0, 255), ("incorrect", 120, 140),
                                        ("incorrect", 230, 255)]):
        d = tmp / "crops" / "test" / cls
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(lo, hi, (40, 40, 3), dtype=np.uint8)).save(d / f"seattle_{i}.png")
    with pytest.warns(UserWarning, match="DynamicQuantizeLinear"):
        df = export_onnx.main(["bench", "--label-type", "X", "--crops", str(tmp / "crops"), "--model", ckpt,
                               "--onnx", f"onnx-int8={int8}", "--threads", "1", "--batch-sizes", "1",
                               "--warmup", "0", "--runs", "1", "--score-batch", "4", "--out", str(tmp / "bench")])
    row = df[df.variant == "onnx-int8"].iloc[0]
    assert row.score_batch == 1 and row.b16_vs_b1_max_abs_dprob > 0

    crops = export_onnx.list_crops(str(tmp / "crops"))
    pixels = export_onnx.preprocess(AutoImageProcessor.from_pretrained(ckpt, use_fast=False), crops.crop_path)
    sess = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    one_by_one = np.concatenate([sess.run(None, {"pixel_values": pixels[i:i + 1]})[0] for i in range(len(pixels))])
    together = sess.run(None, {"pixel_values": pixels})[0]
    assert np.abs(export_onnx.softmax(together) - export_onnx.softmax(one_by_one)).max() > 0   # batch matters here
    scores = pd.read_csv(tmp / "bench" / "scores_X.csv")
    assert scores.label_uid.tolist() == crops.label_uid.tolist()
    np.testing.assert_array_equal(scores["onnx-int8_p_incorrect"].to_numpy(np.float32),
                                  export_onnx.softmax(one_by_one)[:, 1])
