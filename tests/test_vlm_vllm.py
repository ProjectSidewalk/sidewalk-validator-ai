"""Tests for analysis/vlm_vllm.py (issue #63). CPU only, no network, no vllm: the server is a stub ``post``."""
import doctest
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

from analysis import vlm_baseline as vb
from analysis import vlm_vllm as vv


def test_doctests():
    res = doctest.testmod(vv)
    assert res.failed == 0 and res.attempted > 0


def test_no_vllm_import():
    assert "vllm" not in sys.modules


def _resp(pairs, prompt_tokens=700):
    """A canned OpenAI-style chat completion with first-token top logprobs ``pairs`` (token, prob)."""
    top = [{"token": t, "logprob": math.log(p), "bytes": list(t.encode())} for t, p in pairs]
    return {"id": "x", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": pairs[0][0]},
                         "logprobs": {"content": [{"token": pairs[0][0], "logprob": top[0]["logprob"],
                                                   "top_logprobs": top}]}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}}


# ---- score from a canned response ----------------------------------------------------------------------------------

def test_canned_response_matches_full_vocab_score():
    pairs = [("No", 0.6), (" no", 0.1), ("Yes", 0.2), ("yes", 0.05), ("The", 0.03)]
    top, n = vv.parse_chat_response(_resp(pairs))
    s = vv.score_top_logprobs(top)
    # the same distribution as a full vocabulary (rest of the mass on one extra token) through vlm_baseline
    full = np.log(np.array([0.6, 0.1, 0.2, 0.05, 0.03, 0.02]))
    ref = vb.yes_no_score(full, yes_ids=[2, 3], no_ids=[0, 1])
    assert n == 700
    assert s["p_incorrect"] == pytest.approx(ref["p_incorrect"])
    assert s["p_incorrect"] == pytest.approx(0.7 / 0.95)
    assert s["answer_mass"] == pytest.approx(0.95)
    assert s["top20_mass_outside"] == pytest.approx(0.02)
    assert s["answer_floor"] == "" and s["top_token"] == "No"


def test_only_known_spellings_count():
    s = vv.score_top_logprobs([("Yes", math.log(0.5)), ("yes.", math.log(0.3)), ("No", math.log(0.2))])
    assert s["p_yes"] == pytest.approx(0.5) and s["p_no"] == pytest.approx(0.2)


def test_outside_mass_is_never_negative():
    # rounding in returned logprobs can push the visible sum just above 1
    s = vv.score_top_logprobs([("Yes", math.log(0.7) + 1e-9), ("No", math.log(0.3) + 1e-9)])
    assert s["top20_mass_outside"] == 0.0


def test_missing_side_is_floored_and_flagged():
    pairs = [("Yes", 0.9)] + [(f"t{i}", 0.004) for i in range(19)]
    s = vv.score_top_logprobs([(t, math.log(p)) for t, p in pairs])
    assert s["answer_floor"] == "no"
    assert s["p_no"] == pytest.approx(0.004)
    assert s["p_incorrect"] == pytest.approx(0.004 / 0.904)
    assert s["top20_mass_outside"] == pytest.approx(1 - 0.9 - 19 * 0.004)
    both = vv.score_top_logprobs([("The", math.log(0.5)), ("A", math.log(0.1))])
    assert both["answer_floor"] == "both" and both["p_incorrect"] == pytest.approx(0.5)


def test_parse_rejects_response_without_logprobs():
    with pytest.raises(ValueError):
        vv.parse_chat_response({"choices": [{"message": {"content": "No"}}]})


# ---- request shape -------------------------------------------------------------------------------------------------

def test_request_image_first_one_token_top20():
    b = vv.build_request("served", "data:image/webp;base64,AAAA", "Q?", template_kwargs={"enable_thinking": False})
    content = b["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert b["max_tokens"] == 1 and b["temperature"] == 0.0 and b["logprobs"] is True and b["top_logprobs"] == 20
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(b["messages"]) == 1 and "continue_final_message" not in b


def test_prefill_continues_the_assistant_turn():
    b = vv.build_request("served", "data:,", "Q?", prefill="<think></think>")
    assert b["messages"][-1] == {"role": "assistant", "content": "<think></think>"}
    assert b["continue_final_message"] is True and b["add_generation_prompt"] is False


def test_image_data_url(tmp_path):
    p = tmp_path / "a.webp"
    p.write_bytes(b"RIFF1234WEBP")
    assert vv.image_data_url(str(p)).startswith("data:image/webp;base64,UklGRj")


# ---- end to end against a stub server: resumability, columns, timing ---------------------------------------------

def _data_root(tmp_path, n_per=6):
    root = tmp_path / "retrain"
    rows = []
    for t in ("CurbRamp", "Obstacle"):
        (root / t / "correct").mkdir(parents=True)
        (root / t / "incorrect").mkdir(parents=True)
        for i in range(n_per):
            cls = "correct" if i % 2 == 0 else "incorrect"
            split = ("train", "test", "test_heldout")[i % 3]
            crop = f"{cls}/city{i % 2}_{i}.webp"
            (root / t / crop).write_bytes(b"RIFF0000WEBP")
            rows.append({"label_uid": f"city{i % 2}:{t}{i}", "city": f"city{i % 2}", "label_type": t,
                         "label_class": cls, "split": split, "pano_id": f"p{i}", "crop_path": crop, "t": t})
    df = pd.DataFrame(rows)
    for t, g in df.groupby("t"):
        g.drop(columns="t").to_csv(root / t / "split.csv", index=False)
    return root


class _Stub:
    def __init__(self):
        self.calls = []

    def __call__(self, url, body):
        self.calls.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 5, "token_strs": ["<|user|>", "Q", "<|assistant|>", "<think>", "</think>"]}
        q = body["messages"][0]["content"][1]["text"]
        p_no = 0.8 if "obstacle" in q else 0.3
        return _resp([("No", p_no * 0.99), ("Yes", (1 - p_no) * 0.99)])


def _args(root, out, **kw):
    a = ["score", "--model", "stub", "--repo", "org/stub", "--data-root", str(root), "--out-dir", str(out),
         "--label-types", "CurbRamp,Obstacle", "--gpu", "1xTEST", "--concurrency", "2", "--hf-home", str(out)]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a


def test_score_writes_contract_columns_and_resumes(tmp_path, monkeypatch):
    root, out = _data_root(tmp_path), tmp_path / "out"
    stub = _Stub()
    monkeypatch.setattr(vv, "http_post", stub)
    monkeypatch.setattr(vv, "Client", lambda *a, **k: _orig_client(*a, **{**k, "post": stub}))
    monkeypatch.setattr(vv, "_server_version", lambda server: "0.0-test")
    vv.main(_args(root, out, prefill="<think></think>"))
    cr = pd.read_csv(out / "stub" / "CurbRamp.csv", keep_default_na=False)
    assert list(cr.columns) == vv.SCORE_COLUMNS
    assert len(cr) == 4 and cr.label_uid.is_unique   # held-out rows only (4 of 6)
    assert set(cr.batch_size) == {2} and set(cr.gpu) == {"1xTEST"}
    assert cr.p_incorrect.round(6).eq(round(0.3, 6)).all()
    meta = json.load(open(out / "stub" / "CurbRamp.meta.json"))
    assert meta["vllm_version"] == "0.0-test" and meta["request"]["prefill"] == "<think></think>"
    assert meta["rendered_prompt_tail"]["tail"][-1] == "</think>" and meta["n_rows"] == 4
    assert "label_uid" not in json.dumps(meta) and "city0" not in json.dumps(meta)
    n_score_calls = sum(u.endswith("/chat/completions") for u, _ in stub.calls)
    assert n_score_calls == 8
    # second run: nothing left to score, CSV unchanged
    vv.main(_args(root, out, prefill="<think></think>"))
    assert sum(u.endswith("/chat/completions") for u, _ in stub.calls) == n_score_calls
    assert len(pd.read_csv(out / "stub" / "CurbRamp.csv")) == 4
    # a partial CSV is completed without duplicates
    pd.read_csv(out / "stub" / "Obstacle.csv").head(1).to_csv(out / "stub" / "Obstacle.csv", index=False)
    vv.main(_args(root, out))
    ob = pd.read_csv(out / "stub" / "Obstacle.csv")
    assert len(ob) == 4 and ob.label_uid.is_unique
    t = pd.read_csv(out / "stub" / "timing.csv")
    assert list(t.columns) == vv.TIMING_COLUMNS and t.batch_size.tolist() == [2]


_orig_client = vv.Client


def test_limit_rows_takes_half_per_class():
    df = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(10)], "label_type": "CurbRamp",
                       "label_class": ["correct", "incorrect"] * 5})
    assert vv.limit_rows(df, 4).label_class.value_counts().to_dict() == {"correct": 2, "incorrect": 2}
    assert len(vv.limit_rows(df, 0)) == 10


def test_upsert_timing_replaces_same_key(tmp_path):
    p = str(tmp_path / "timing.csv")
    vv.upsert_timing(p, vv.timing_row("m", "4xH200", 1, [1.0, 2.0, 3.0], "a"))
    vv.upsert_timing(p, vv.timing_row("m", "4xH200", 32, [0.1], "b"))
    vv.upsert_timing(p, vv.timing_row("m", "4xH200", 1, [5.0], "c"))
    t = pd.read_csv(p).sort_values("batch_size")
    assert t.batch_size.tolist() == [1, 32] and t.source.tolist() == ["c", "b"]


def test_client_retries_then_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vv.time, "sleep", lambda s: None)
    img = tmp_path / "a.webp"
    img.write_bytes(b"x")
    calls = []

    def flaky(url, body):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("connection reset")
        return _resp([("Yes", 0.9), ("No", 0.1)])

    assert vv.Client("http://s", "m", post=flaky).score(str(img), "Q?")["p_incorrect"] == pytest.approx(0.1)
    with pytest.raises(OSError):
        vv.Client("http://s", "m", post=lambda u, b: (_ for _ in ()).throw(OSError("down"))).score(str(img), "Q?")


def test_hf_revision(tmp_path):
    d = tmp_path / "hub" / "models--org--m" / "refs"
    d.mkdir(parents=True)
    (d / "main").write_text("abc123\n")
    assert vv.hf_revision("org/m", str(tmp_path)) == "abc123"
    assert vv.hf_revision("org/other", str(tmp_path)) is None
    assert os.path.isdir(tmp_path)


def test_refuses_another_backends_files(tmp_path, monkeypatch):
    """A transformers-leg CSV or meta in the same <out>/<model>/ is never appended to or merged into."""
    root, out = _data_root(tmp_path), tmp_path / "out"
    stub = _Stub()
    monkeypatch.setattr(vv, "Client", lambda *a, **k: _orig_client(*a, **{**k, "post": stub}))
    monkeypatch.setattr(vv, "_server_version", lambda server: "0.0-test")
    d = out / "stub"
    d.mkdir(parents=True)
    other = pd.DataFrame({"label_uid": ["city0:CurbRamp1"], "p_incorrect": [0.2], "batch_size": [16], "gpu": ["A40"]})
    other.to_csv(d / "CurbRamp.csv", index=False)
    with pytest.raises(SystemExit):
        vv.main(_args(root, out))
    assert pd.read_csv(d / "CurbRamp.csv").equals(other) and not stub.calls
    (d / "CurbRamp.csv").unlink()
    meta = {"model": "stub", "gpu": "NVIDIA A40", "batch_size": 16}
    (d / "CurbRamp.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit):
        vv.main(_args(root, out))
    assert json.loads((d / "CurbRamp.meta.json").read_text()) == meta
