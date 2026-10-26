"""Tests for the issue #63 torch path of analysis/vlm_baseline.py. CPU only, no torch, no weights, no network."""
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from analysis import vlm_baseline as vb


# ---- padding and batch assembly ------------------------------------------------------------------------------------

def _stub_forward(ids, mask):
    """A causal stub "model": the logits at position t depend only on the real tokens up to t (padding ignored), so
    the next-token logits of a row are a function of that row's own tokens."""
    v = 7
    out = np.zeros(ids.shape + (v,))
    for b in range(ids.shape[0]):
        real = [int(x) for x, m in zip(ids[b], mask[b]) if m]
        k = 0
        for t in range(ids.shape[1]):
            if mask[b, t]:
                k += 1
                h = sum((i + 1) * x for i, x in enumerate(real[:k]))
                out[b, t] = np.cos(np.arange(v) * (h % 11 + 1))
    return out


def _pad(seqs, side):
    t = max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), t), int)
    mask = np.zeros((len(seqs), t), int)
    for i, s in enumerate(seqs):
        if side == "left":
            ids[i, t - len(s):], mask[i, t - len(s):] = s, 1
        else:
            ids[i, :len(s)], mask[i, :len(s)] = s, 1
    return ids, mask


@pytest.mark.parametrize("side", ["left", "right"])
def test_batch_and_single_give_the_same_next_token_logits(side):
    seqs = [[3, 1, 4, 1, 5], [9, 2], [6, 5, 3, 5]]
    ids, mask = _pad(seqs, side)
    logits = _stub_forward(ids, mask)
    pos = vb.last_token_positions(mask)
    batched = logits[np.arange(len(seqs)), pos]
    for i, s in enumerate(seqs):
        one_ids, one_mask = _pad([s], side)
        alone = _stub_forward(one_ids, one_mask)[0, -1]
        np.testing.assert_allclose(batched[i], alone)
    if side == "left":
        assert (pos == ids.shape[1] - 1).all()   # with left padding the last column is every row's last token
    s_b = vb.score_rows(batched, [0, 1], [2, 3])
    s_1 = [vb.score_rows(_stub_forward(*_pad([s], side))[:, -1], [0, 1], [2, 3])[0] for s in seqs]
    assert [r["p_incorrect"] for r in s_b] == pytest.approx([r["p_incorrect"] for r in s_1])


def test_score_rows_matches_yes_no_score_on_log_softmax():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 10)) * 4
    rows = vb.score_rows(x, [0, 1], [2], decode=lambda ids: f"t{ids[0]}")
    for r, xi in zip(rows, x):
        lp = xi - np.log(np.exp(xi).sum())
        assert r["p_incorrect"] == pytest.approx(vb.yes_no_score(lp, [0, 1], [2])["p_incorrect"])
        assert r["p_yes"] == pytest.approx(np.exp(lp[[0, 1]]).sum())
        assert r["top_token"] == f"t{int(xi.argmax())}"


# ---- yes/no ids across tokenizers ----------------------------------------------------------------------------------

def test_answer_ids_bpe_with_leading_space_tokens():
    """GPT-2 style BPE: " yes" is its own token (Ġyes), distinct from "yes"."""
    vocab = {"yes": [1], "Yes": [2], "YES": [3], " yes": [4], " Yes": [5], " YES": [6, 7],
             "no": [10], "No": [11], "NO": [12], " no": [13], " No": [14], " NO": [15]}
    y, yu = vb.answer_token_ids(lambda s: vocab[s], vb.YES_VARIANTS)
    n, nu = vb.answer_token_ids(lambda s: vocab[s], vb.NO_VARIANTS)
    assert y == [1, 2, 3, 4, 5] and " YES" not in yu
    assert n == [10, 11, 12, 13, 14, 15]


def test_answer_ids_sentencepiece_without_leading_space_variants():
    """SentencePiece with a dummy prefix: "yes" and " yes" both encode to ▁yes; "YES" splits. Deduplicated."""
    vocab = {"yes": [50], "Yes": [51], "YES": [52, 53], " yes": [50], " Yes": [51], " YES": [52, 53],
             "no": [60], "No": [61], "NO": [62], " no": [60], " No": [61], " NO": [62]}
    y, _ = vb.answer_token_ids(lambda s: vocab[s], vb.YES_VARIANTS)
    n, _ = vb.answer_token_ids(lambda s: vocab[s], vb.NO_VARIANTS)
    assert y == [50, 51] and n == [60, 61, 62]


# ---- thinking switch -----------------------------------------------------------------------------------------------

def test_thinking_switch_only_for_families_that_have_it():
    for short, spec in vb.TORCH_MODELS.items():
        kw = vb.chat_kwargs(short)
        extra = spec.get("template_kwargs") or {}
        if spec["thinking"]:
            assert kw == {**extra, spec["thinking"]: False}, short
        else:
            assert kw == extra and False not in kw.values(), short
    assert vb.chat_kwargs("qwen3.8-27b") == {"enable_thinking": False}
    assert vb.chat_kwargs("gemma4-31b") == {"enable_thinking": False}
    assert vb.chat_kwargs("qwen3vl-8b-bf16") == {}            # instruct model, no switch in its template
    assert vb.chat_kwargs("molmo2-8b") == {}
    assert vb.chat_kwargs("muse-glimmer-30b") == {"reasoning_strength": "low"}


def test_answer_prefix_only_where_the_model_chooses_its_own_mode():
    assert vb.with_answer_prefix("P", "qwen3.8-27b") == "P"
    assert vb.with_answer_prefix("<|start|>assistant", "muse-glimmer-30b") == "<|start|>assistant to=user<|message|>"
    assert vb.with_answer_prefix("x", "phi4-rv-15b") == "x<nothink>"


def test_roster_names_are_distinct_from_the_mlx_legs():
    assert not set(vb.TORCH_MODELS) & set(vb.MODELS)
    assert all(spec["repo"].count("/") == 1 for spec in vb.TORCH_MODELS.values())


# ---- few-shot ------------------------------------------------------------------------------------------------------

def _train(n=40):
    return pd.DataFrame({"label_uid": [f"c:{i}" for i in range(n)], "pano_id": [f"p{i // 2}" for i in range(n)],
                         "label_class": ["correct", "incorrect"] * (n // 2), "image": [f"/x/{i}.webp" for i in range(n)]})


def test_fewshot_set_is_fixed_balanced_and_seeded():
    pools = vb.fewshot_pools(_train())
    a = vb.fewshot_examples(pools, None, 8)
    b = vb.fewshot_examples(vb.fewshot_pools(_train()), "unrelated-pano", 8)
    assert a.label_uid.tolist() == b.label_uid.tolist()            # shared prefix for unrelated queries
    assert a.label_class.tolist() == ["correct", "incorrect"] * 4
    c = vb.fewshot_examples(vb.fewshot_pools(_train(), seed=1), None, 8)
    assert c.label_uid.tolist() != a.label_uid.tolist()


def test_fewshot_excludes_the_query_panorama_by_swapping_in_the_next():
    pools = vb.fewshot_pools(_train())
    base = vb.fewshot_examples(pools, None, 8)
    clash = base.pano_id.iloc[0]
    swapped = vb.fewshot_examples(pools, clash, 8)
    assert clash not in set(swapped.pano_id)
    assert len(swapped) == 8 and (swapped.label_class == "correct").sum() == 4
    kept = [u for u in base.label_uid if base.set_index("label_uid").pano_id[u] != clash]
    assert set(kept) <= set(swapped.label_uid)                      # only the clashing examples change


def test_fewshot_messages_put_examples_before_the_query():
    ex = [("img-a", vb.answer_word("correct")), ("img-b", vb.answer_word("incorrect"))]
    m = vb.build_messages("Is it?", "query", ex)
    assert [c["content"][0].get("image") for c in m if c["role"] == "user"] == ["img-a", "img-b", "query"]
    assert [c["content"][0]["text"] for c in m if c["role"] == "assistant"] == ["Yes", "No"]
    assert vb.build_messages("Is it?", "q")[0]["content"][0] == {"type": "image", "image": "q"}
    with pytest.raises(ValueError):
        vb.fewshot_examples(vb.fewshot_pools(_train()), None, 7)


# ---- resumability and the score-torch loop with a stub model -------------------------------------------------------

def _split_root(tmp_path, n=12):
    from PIL import Image
    root = tmp_path / "run"
    for t in ("CurbRamp",):
        rows = []
        for i in range(n):
            cls = "correct" if i % 2 else "incorrect"
            split = "train" if i >= n - 4 else ("test" if i % 3 else "test_heldout")
            p = f"{cls}/city_{i}.webp"
            (root / t / cls).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (i, 0, 0)).save(root / t / p)
            rows.append({"label_uid": f"city:{i}", "city": "city", "label_type": t, "label_class": cls,
                         "split": split, "pano_id": f"p{i}", "crop_path": p, "status": "ok"})
        pd.DataFrame(rows).to_csv(root / t / "split.csv", index=False)
    return root


class _StubVlm:
    """Stands in for TorchVlm: scores from the number of images in each conversation and its label uid hash."""
    calls = []

    def __init__(self, short):
        self.repo, self.loader, self.device_map = "stub/stub", "stub", "cpu"
        self.yes_used, self.yes_ids, self.no_used, self.no_ids = ["yes"], [0], ["no"], [1]

    def score_batch(self, convs):
        _StubVlm.calls.append(len(convs))
        out = []
        for c in convs:
            n_img = sum(1 for m in c if m["role"] == "user")
            out.append({"p_yes": .5, "p_no": .5, "answer_mass": 1.0, "log_odds_incorrect": 0.0,
                        "p_incorrect": 0.1 * n_img, "p_correct": 1 - 0.1 * n_img, "top_token": "yes",
                        "prompt_tokens": 10 * n_img})
        return out, 0.01 * len(convs)


def _args(root, out, **kw):
    a = dict(model="qwen3vl-8b-bf16", split_root=str(root), out_dir=str(out), variant="definition",
             label_types="CurbRamp", limit=0, batch=3, chunk=2, fewshot=0, max_batch_diff=0.02)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_score_torch_writes_mlx_compatible_columns_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "TorchVlm", _StubVlm)
    monkeypatch.setattr(vb, "hf_snapshot", lambda repo: (None, None))
    root = _split_root(tmp_path)
    out = tmp_path / "out"
    vb.run_score_torch(_args(root, out, limit=4))
    path = out / "scores" / "qwen3vl-8b-bf16" / "CurbRamp.csv"
    first = pd.read_csv(path)
    assert len(first) == 4
    assert list(first.columns)[:12] == ["label_uid", "model", "variant", "p_yes", "p_no", "answer_mass",
                                       "log_odds_incorrect", "p_incorrect", "p_correct", "top_token",
                                       "prompt_tokens", "latency_s"]
    assert list(first.columns)[-2:] == ["batch_size", "gpu"]
    _StubVlm.calls.clear()
    vb.run_score_torch(_args(root, out))                     # full run: only the rows not yet scored
    full = pd.read_csv(path)
    assert full.label_uid.is_unique and len(full) == 8     # 8 held-out rows, train excluded
    assert sum(_StubVlm.calls) == 4                        # only the 4 new rows (batch check done on run 1)
    assert vb.batch_invariance(str(out / "scores"), str(out / "timing")).empty
    meta = json.loads((out / "scores" / "qwen3vl-8b-bf16" / "CurbRamp.meta.json").read_text())
    assert meta["batch_check"]["within_1e-3"] and meta["batch_check"]["n"] == 3 and len(meta["runs"]) == 2
    vb.run_score_torch(_args(root, out))                     # nothing left: the model is never loaded
    assert len(pd.read_csv(path)) == 8


def test_score_torch_fewshot_records_examples_and_adds_example_images(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "TorchVlm", _StubVlm)
    monkeypatch.setattr(vb, "hf_snapshot", lambda repo: (None, None))
    root = _split_root(tmp_path, n=20)
    out = tmp_path / "out"
    vb.run_score_torch(_args(root, out, fewshot=2))
    leg = out / "scores" / "qwen3vl-8b-bf16-fewshot2"
    d = pd.read_csv(leg / "CurbRamp.csv")
    assert np.allclose(d.p_incorrect, 0.3)                    # 2 examples + the query
    meta = json.loads((leg / "CurbRamp.meta.json").read_text())
    assert len(meta["fewshot_examples"]) == 2 and meta["fewshot_seed"] == vb.FEWSHOT_SEED
    held = set(pd.read_csv(root / "CurbRamp" / "split.csv").query("split != 'train'").label_uid)
    assert not set(meta["fewshot_examples"]) & held


def test_todo_rows_and_pilot_rows(tmp_path):
    df = pd.DataFrame({"label_uid": ["a", "b", "c"], "label_class": ["correct", "incorrect", "correct"]})
    p = tmp_path / "s.csv"
    assert len(vb.todo_rows(df, str(p))) == 3
    pd.DataFrame({"label_uid": ["b"]}).to_csv(p, index=False)
    assert vb.todo_rows(df, str(p)).label_uid.tolist() == ["a", "c"]
    assert set(vb.pilot_rows(df, 2).label_class) == {"correct", "incorrect"}


def test_export_heldout_and_timing_summary(tmp_path):
    root = _split_root(tmp_path)
    for t in vb.LABEL_TYPES:
        if t != "CurbRamp":
            os.makedirs(root / t, exist_ok=True)
            pd.read_csv(root / "CurbRamp" / "split.csv").assign(label_type=t).to_csv(root / t / "split.csv",
                                                                                     index=False)
    for t in vb.LABEL_TYPES:
        s = pd.read_csv(root / t / "split.csv")
        pd.DataFrame({"label_uid": s.label_uid, "split": s.split, "true": s.label_class,
                      "p_correct": .4, "p_incorrect": .6}).to_csv(root / t / "predictions.csv", index=False)
    out = tmp_path / "stage"
    vb.run_export_heldout(types.SimpleNamespace(split_root=str(root), out_dir=str(out)))
    sp = pd.read_csv(out / "splits" / "Obstacle.csv")
    assert list(sp.columns) == vb.SPLIT_COLUMNS and set(sp.split) <= set(vb.HELDOUT_SPLITS)
    r3 = pd.read_csv(out / "scores" / "round3" / "Obstacle.csv")
    assert list(r3.columns) == ["label_uid", "p_incorrect"] and len(r3) == len(sp)
    leg = out / "scores" / "m"
    leg.mkdir()
    pd.DataFrame({"label_uid": ["x", "y", "z"], "latency_s": [.1, .2, .3], "batch_size": [8, 8, 2],
                  "gpu": "NVIDIA L40S"}).to_csv(leg / "CurbRamp.csv", index=False)
    t = vb.timing_summary(str(out / "scores"), str(out / "timing"))
    assert t.to_dict("records") == [{"model": "m", "gpu": "NVIDIA L40S", "batch_size": 8, "n_crops": 2,
                                     "median_s_per_crop": pytest.approx(.15), "p90_s_per_crop": pytest.approx(.19),
                                     "source": "full run"}]


def test_patch_cuda_prod_matches_prod_semantics():
    """The cumprod route that patch_cuda_prod uses gives prod's values and shapes (on CPU; skipped without torch)."""
    torch = pytest.importorskip("torch")
    x = torch.tensor([[1, 22, 34], [2, 20, 30]])
    for dim, keep in ((-1, False), (0, True), (None, False)):
        ref = x.prod() if dim is None else x.prod(dim, keepdim=keep)
        y = x if dim is None else x
        d = 0 if dim is None else dim
        y = y.reshape(-1) if dim is None else y
        alt = y.cumprod(d).narrow(d, y.shape[d] - 1, 1)
        alt = alt if keep else alt.squeeze(d)
        assert torch.equal(ref, alt)


def test_batch_invariance_compares_timing_rows_with_the_run(tmp_path):
    leg = tmp_path / "scores" / "m"
    leg.mkdir(parents=True)
    pd.DataFrame({"label_uid": ["a", "b"], "p_incorrect": [.5, .9], "log_odds_incorrect": [0, 2.2],
                  "batch_size": 16, "gpu": "NVIDIA L40S"}).to_csv(leg / "T.csv", index=False)
    (tmp_path / "timing").mkdir()
    (tmp_path / "timing" / "m__nvidia-a40.meta.json").write_text('{"gpu": "NVIDIA A40"}')
    pd.DataFrame({"label_uid": ["a", "b"], "latency_s": .1, "p_incorrect": [.52, .9], "gpu": "NVIDIA A40",
                  "log_odds_incorrect": [.08, 2.2]}).to_csv(tmp_path / "timing" / "m__nvidia-a40.csv", index=False)
    b = vb.batch_invariance(str(tmp_path / "scores"), str(tmp_path / "timing"))
    assert b.model.tolist() == ["m"] and b.n.tolist() == [2] and b.max_abs_dp.iloc[0] == pytest.approx(.02)
    assert b.timing_gpu.tolist() == ["NVIDIA A40"] and b.run_gpu.tolist() == ["NVIDIA L40S"]
    assert vb.timing_summary(str(tmp_path / "scores"), str(tmp_path / "timing")).model.tolist()[0] == "m"


def test_timing_summary_uses_a_legs_own_timing_csv(tmp_path):
    """The vLLM leg's per-request latency_s is wall time under concurrency; its own timing.csv must win."""
    leg = tmp_path / "scores" / "glm"
    leg.mkdir(parents=True)
    pd.DataFrame({"label_uid": ["a", "b"], "latency_s": [3.2, 3.2], "batch_size": 32,
                  "gpu": "4xNVIDIA H200"}).to_csv(leg / "CurbRamp.csv", index=False)
    own = pd.DataFrame([{"model": "glm", "gpu": "4xNVIDIA H200", "batch_size": 32, "n_crops": 2,
                         "median_s_per_crop": 0.1, "p90_s_per_crop": 0.12, "source": "vllm chunk wall"}])
    own.to_csv(leg / "timing.csv", index=False)
    t = vb.timing_summary(str(tmp_path / "scores"), str(tmp_path / "timing"))
    assert t.to_dict("records") == own.to_dict("records")
