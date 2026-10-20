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
