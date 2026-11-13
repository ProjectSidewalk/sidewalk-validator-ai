"""Tests for analysis/vlm_gemini.py (issue #63, the Gemini leg). CPU only, no network: ``post`` is a stub, and the
google-genai SDK is never imported except by the one adapter test that skips without it."""
import doctest
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import vlm_baseline as vb
from analysis import vlm_gemini as vg

LISTED, UNLISTED = "pub-a", "other-z"     # synthetic deployment ids


def test_doctests():
    res = doctest.testmod(vg)
    assert res.failed == 0 and res.attempted > 0


# ---- the score ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("answer,conf,p", [("no", 80, 0.8), ("yes", 80, 0.2), ("no", 0, 0.0), ("yes", 0, 1.0),
                                           ("no", 100, 1.0), ("yes", 100, 0.0), ("YES", 50, 0.5)])
def test_score_mapping(answer, conf, p):
    s = vg.score_response({"text": json.dumps({"answer": answer, "confidence": conf}), "finish_reason": "STOP"})
    assert s["p_incorrect"] == pytest.approx(p) and s["p_correct"] == pytest.approx(1 - p)
    assert s["unscored_reason"] == "" and s["score_source"] == "self_reported"
    assert all(s[c] is None for c in vg.LOGPROB_COLUMNS)


@pytest.mark.parametrize("text,reason", [
    (None, "empty answer"), ("  ", "empty answer"), ("No.", "not JSON"), ("[1]", "not a JSON object"),
    ('{"answer": "maybe", "confidence": 50}', "answer not yes/no"), ('{"confidence": 50}', "answer not yes/no"),
    ('{"answer": "yes"}', "confidence not a number"),
    ('{"answer": "yes", "confidence": "80"}', "confidence not a number"),
    ('{"answer": "yes", "confidence": true}', "confidence not a number"),
    ('{"answer": "no", "confidence": 100.5}', "confidence outside 0-100"),
    ('{"answer": "no", "confidence": -1}', "confidence outside 0-100")])
def test_malformed_reply_is_unscored_never_half(text, reason):
    s = vg.score_response({"text": text, "finish_reason": "STOP"})
    assert s["p_incorrect"] is None and s["p_correct"] is None and s["unscored_reason"] == reason


def test_unscored_reason_names_an_abnormal_finish():
    s = vg.score_response({"text": '{"answer": "ye', "finish_reason": "MAX_TOKENS"})
    assert s["unscored_reason"] == "not JSON (finish_reason MAX_TOKENS)" and s["finish_reason"] == "MAX_TOKENS"


# ---- the request --------------------------------------------------------------------------------------------------

def test_request_is_build_prompt_unchanged_plus_the_fixed_sentence():
    for t in vb.LABEL_TYPES:
        for v in vb.VARIANTS:
            r = vg.build_request("gemini-x", t, "/c/q.webp", v, thinking_level="minimal")
            (turn,) = r["contents"]
            assert turn["role"] == "user" and turn["parts"][0] == {"image": "/c/q.webp"}   # image first
            assert turn["parts"][1]["text"] == vb.build_prompt(t, v) + " " + vg.JSON_SENTENCE
    c = r["config"]
    assert c["temperature"] == 0.0 and c["response_mime_type"] == "application/json"
    assert c["response_schema"] == vg.RESPONSE_SCHEMA and c["thinking_level"] == "minimal"
    assert vg.RESPONSE_SCHEMA["properties"]["answer"]["enum"] == ["yes", "no"]


def test_retryable_errors():
    class ApiError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.code = code
    assert all(vg.is_retryable(ApiError(c)) for c in (408, 429, 500, 503))
    assert not any(vg.is_retryable(e) for e in (ApiError(400), ApiError(403), FileNotFoundError("x"), ValueError()))


def test_cost_and_cities_file(tmp_path):
    assert vg.estimate_cost_usd("gemini-3.8-flash", 2_000_000, 0, 0) == pytest.approx(1.5)
    assert vg.estimate_cost_usd("gemini-unknown", 10, 10, 10) is None
    (tmp_path / "a.json").write_text(json.dumps(["b", "a"]))
    (tmp_path / "b.txt").write_text("# public\na\n\nb\n")
    (tmp_path / "c.json").write_text(json.dumps({"cities": [{"city_id": "a", "visibility": "public"},
                                                            {"city_id": "b", "visibility": "public"},
                                                            {"city_id": "z", "visibility": "private"}]}))
    got = [vg.load_cities_file(str(tmp_path / f)) for f in ("a.json", "b.txt", "c.json")]
    assert got[0] == got[1] == got[2] and got[0][0] == ["a", "b"]


# ---- end to end against a stub ``post`` ---------------------------------------------------------------------------

def _data_root(tmp_path, abs_prefix=None):
    """Two types x two deployments x {train, test, test_heldout} x two classes x 6 rows. Row j of a (city, class)
    has panorama ``<city>-<class>-<j>`` in every split, so each query shares a panorama with one train row."""
    root = tmp_path / "round3"
    rows = []
    for t in ("CurbRamp", "Obstacle"):
        for city in (LISTED, UNLISTED):
            for split in ("train", "test", "test_heldout"):
                for cls in ("correct", "incorrect"):
                    for j in range(6):
                        crop = f"{cls}/{city}_{split}_{j}.webp"
                        (root / t / cls).mkdir(parents=True, exist_ok=True)
                        (root / t / crop).write_bytes(b"RIFF0000WEBP")
                        rows.append({"label_uid": f"{city}:{t}-{split}-{cls}-{j}", "city": city, "label_type": t,
                                     "label_class": cls, "split": split, "pano_id": f"{city}-{cls}-{j}",
                                     "crop_path": f"{abs_prefix}/{t}/{crop}" if abs_prefix else crop, "status": "ok"})
    df = pd.DataFrame(rows)
    for t, g in df.groupby("label_type"):
        g.to_csv(root / t / "split.csv", index=False)
    return root, df


class _Stub:
    """Answers "no" at 80 for Obstacle and "yes" at 70 for CurbRamp; ``bad`` query images get a non-JSON reply."""

    def __init__(self, bad=(), fail=None):
        self.requests, self.bad, self.fail = [], set(bad), fail

    def __call__(self, request):
        self.requests.append(request)
        if self.fail:
            self.fail(request)
        q = request["contents"][-1]
        if os.path.basename(q["parts"][0]["image"]) in self.bad:
            return {"text": "Yes.", "finish_reason": "STOP", "usage": {"prompt": 300, "output": 2}}
        no = "obstacle" in q["parts"][1]["text"]
        return {"text": json.dumps({"answer": "no" if no else "yes", "confidence": 80 if no else 70}),
                "finish_reason": "STOP", "model_version": "gemini-x-001",
                "usage": {"prompt": 300, "output": 10, "thinking": 5, "cached": 0, "total": 315}}


def _args(root, tmp_path, **kw):
    cities = tmp_path / "cities.txt"
    cities.write_text(LISTED + "\n")
    a = {"model": "gemini-3.8-flash", "data-root": root, "out-dir": tmp_path / "scores", "cities-file": cities,
         "usage-log": tmp_path / "usage.jsonl", "label-types": "CurbRamp,Obstacle", "concurrency": 2, **kw}
    return ["score"] + [x for k, v in a.items() for x in (f"--{k}", str(v))]


def test_scores_listed_deployments_only_and_resumes(tmp_path):
    root, split = _data_root(tmp_path)
    stub = _Stub(bad={f"{LISTED}_test_0.webp"})
    vg.main(_args(root, tmp_path), post=stub)
    leg = tmp_path / "scores" / "gemini-3.8-flash"
    cr = pd.read_csv(leg / "CurbRamp.csv")
    assert list(cr.columns) == vg.SCORE_COLUMNS
    want = split[(split.label_type == "CurbRamp") & (split.city == LISTED) & split.split.isin(vb.HELDOUT_SPLITS)]
    assert sorted(cr.label_uid) == sorted(want.label_uid) and cr.label_uid.str.startswith(LISTED + ":").all()
    assert not any(UNLISTED in json.dumps(r) for r in stub.requests)          # nothing of the other deployment sent
    assert cr[list(vg.LOGPROB_COLUMNS)].isna().all().all() and set(cr.score_source) == {"self_reported"}
    bad = cr[cr.label_uid.str.contains(r"-test-(?:in)?correct-0$")]
    assert len(bad) == 2 and bad.p_incorrect.isna().all() and (bad.unscored_reason == "not JSON").all()
    good = cr.drop(bad.index)
    assert good.p_incorrect.round(6).eq(0.3).all() and good.unscored_reason.isna().all()
    meta = json.load(open(leg / "CurbRamp.meta.json"))
    assert meta["backend"] == vg.BACKEND and meta["model"] == "gemini-3.8-flash" and meta["temperature"] == 0.0
    assert meta["json_sentence"] == vg.JSON_SENTENCE and meta["response_schema"] == vg.RESPONSE_SCHEMA
    assert meta["thinking_level"] == "minimal" and meta["score_source"] == "self_reported"
    assert meta["prompt"] == vb.build_prompt("CurbRamp") and meta["n_cities"] == 1
    assert meta["n_crops_listed"] == 24 and meta["n_crops_heldout"] == 48 and len(meta["cities_sha256"]) == 64
    assert meta["finish_reasons"] == {"STOP": 24} and meta["n_unscored"] == 2 and meta["runs"][0]["n_failed"] == 0
    assert UNLISTED not in json.dumps(meta) and "label_uid" not in json.dumps(meta)
    usage = [json.loads(x) for x in open(tmp_path / "usage.jsonl")]
    assert [u["label_type"] for u in usage] == ["CurbRamp", "Obstacle"]
    u = usage[1]
    # 22 scored replies (300 in, 10 out, 5 thinking) and 2 unscored ones (300 in, 2 out), Obstacle as CurbRamp
    assert (u["n_calls"], u["n_scored"], u["n_unscored"], u["n_failed"]) == (24, 22, 2, 0)
    assert (u["input_tokens"], u["output_tokens"], u["thinking_tokens"]) == (24 * 300, 22 * 10 + 2 * 2, 22 * 5)
    assert u["est_cost_usd"] == pytest.approx((24 * 300 * 0.75 + (22 * 15 + 2 * 2) * 3.75) / 1e6)
    assert u["model_versions"] == ["gemini-x-001"] and u["leg"] == "gemini-3.8-flash"
    # resume: nothing left, nothing sent, no ledger line
    n = len(stub.requests)
    vg.main(_args(root, tmp_path), post=stub)
    assert len(stub.requests) == n and len(open(tmp_path / "usage.jsonl").readlines()) == 2
    # a partial CSV is completed without duplicates
    pd.read_csv(leg / "Obstacle.csv").head(5).to_csv(leg / "Obstacle.csv", index=False)
    vg.main(_args(root, tmp_path), post=stub)
    ob = pd.read_csv(leg / "Obstacle.csv")
    assert len(stub.requests) == n + 19 and len(ob) == 24 and ob.label_uid.is_unique
    # other settings into the same leg are refused
    with pytest.raises(SystemExit, match="thinking_level"):
        vg.main(_args(root, tmp_path, **{"thinking-level": "high"}), post=stub)


def test_limit_takes_half_per_class_per_type(tmp_path):
    root, _ = _data_root(tmp_path)
    stub = _Stub()
    vg.main(_args(root, tmp_path, limit=4), post=stub)
    for t in ("CurbRamp", "Obstacle"):
        d = pd.read_csv(tmp_path / "scores" / "gemini-3.8-flash" / f"{t}.csv")
        cls = d.label_uid.str.extract(r"-(correct|incorrect)-")[0]
        assert cls.value_counts().to_dict() == {"correct": 2, "incorrect": 2}
    assert len(stub.requests) == 8


def test_fewshot_examples_are_listed_off_pano_and_match_the_open_weight_rule(tmp_path):
    root, split = _data_root(tmp_path)
    stub = _Stub()
    vg.main(_args(root, tmp_path, fewshot=4, **{"label-types": "CurbRamp"}), post=stub)
    img = {os.path.join(str(root), "CurbRamp", r.crop_path): r
           for r in split[split.label_type == "CurbRamp"].itertuples()}
    q = vb.build_prompt("CurbRamp")
    train = vg.with_city(vb.load_split_rows(str(root), "CurbRamp", ("train",)))
    pools = vg.listed_pools(train, [LISTED])
    for r in stub.requests:
        *ex, query = r["contents"]
        qrow = img[query["parts"][0]["image"]]
        users, answers = ex[0::2], ex[1::2]
        rows = [img[u["parts"][0]["image"]] for u in users]
        assert len(rows) == 4 and all(x.city == LISTED and x.split == "train" for x in rows)
        assert qrow.pano_id not in {x.pano_id for x in rows}
        assert all(u["parts"][1]["text"] == q for u in users)                  # examples: question unchanged
        assert [a["parts"][0]["text"] for a in answers] == [vb.answer_word(x.label_class) for x in rows]
        assert query["parts"][1]["text"] == q + " " + vg.JSON_SENTENCE
        assert [x.label_uid for x in rows] == vb.fewshot_examples(pools, qrow.pano_id, 4).label_uid.tolist()
    meta = json.load(open(tmp_path / "scores" / "gemini-3.8-flash-fewshot4" / "CurbRamp.meta.json"))
    assert meta["fewshot"] == 4 and meta["fewshot_seed"] == vb.FEWSHOT_SEED
    assert isinstance(meta["fewshot_matches_open_weight_arm"], bool)
    assert all(u.startswith(LISTED + ":") for u in meta["fewshot_examples"])


def test_refuses_a_git_checkout_for_outputs_and_ledger(tmp_path):
    root, _ = _data_root(tmp_path)
    inside = os.path.join(vg.REPO_DIR, "scores-should-not-exist")
    stub = _Stub()
    with pytest.raises(SystemExit, match="--out-dir"):
        vg.main(_args(root, tmp_path, **{"out-dir": inside}), post=stub)
    with pytest.raises(SystemExit, match="--usage-log"):
        vg.main(_args(root, tmp_path, **{"usage-log": os.path.join(inside, "usage.jsonl")}), post=stub)
    assert not os.path.exists(inside) and not stub.requests
    vg.main(_args(root, tmp_path, **{"usage-log": "none", "limit": 2}), post=stub)
    assert stub.requests and not (tmp_path / "usage.jsonl").exists()


def test_retries_then_records_failures_for_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "backoff_s", lambda attempt: 0.0)

    class ApiError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.code = code
    root, _ = _data_root(tmp_path)
    calls = {}

    def flaky(req):     # each crop: 429 twice, then fine; one crop always 503
        key = req["contents"][-1]["parts"][0]["image"]
        calls[key] = calls.get(key, 0) + 1
        if key.endswith(f"{LISTED}_test_1.webp") or calls[key] <= 2:
            raise ApiError(503 if key.endswith("_test_1.webp") else 429)
    stub = _Stub(fail=flaky)
    vg.main(_args(root, tmp_path, retries=2, limit=0, **{"label-types": "CurbRamp"}), post=stub)
    cr = pd.read_csv(tmp_path / "scores" / "gemini-3.8-flash" / "CurbRamp.csv")
    assert len(cr) == 22 and not cr.label_uid.str.contains(r"-test-(?:in)?correct-1$").any()
    u = json.loads(open(tmp_path / "usage.jsonl").read())
    assert u["n_failed"] == 2 and u["n_calls"] == 22 and u["n_attempts"] == 22 * 3 + 2 * 3
    stub.fail = None                                    # the next run retries only the two failures
    vg.main(_args(root, tmp_path, **{"label-types": "CurbRamp"}), post=stub)
    assert len(pd.read_csv(tmp_path / "scores" / "gemini-3.8-flash" / "CurbRamp.csv")) == 24


def test_non_retryable_error_stops_after_writing_the_good_rows(tmp_path):
    class ApiError(Exception):
        code = 400
    root, _ = _data_root(tmp_path)

    def bad_one(req):
        if req["contents"][-1]["parts"][0]["image"].endswith(f"{LISTED}_test_2.webp"):
            raise ApiError("invalid argument")
    with pytest.raises(SystemExit, match="non-retryable"):
        vg.main(_args(root, tmp_path, chunk=100, **{"label-types": "CurbRamp"}), post=_Stub(fail=bad_one))
    assert len(pd.read_csv(tmp_path / "scores" / "gemini-3.8-flash" / "CurbRamp.csv")) == 22


def test_crop_root_rewrite(tmp_path):
    root, _ = _data_root(tmp_path, abs_prefix="/elsewhere/round3")
    stub = _Stub()
    with pytest.raises(SystemExit, match="missing"):
        vg.main(_args(root, tmp_path, limit=2), post=stub)
    vg.main(_args(root, tmp_path, limit=2, **{"crop-root-from": "/elsewhere/round3", "crop-root-to": root}),
            post=stub)
    assert all(r["contents"][-1]["parts"][0]["image"].startswith(str(root)) for r in stub.requests)


# ---- the SDK adapter ------------------------------------------------------------------------------------------------

def test_from_sdk_response_reads_text_finish_and_usage():
    part = SimpleNamespace(text='{"answer": "no", "confidence": 90}', thought=False)
    thought = SimpleNamespace(text="hmm", thought=True)
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"),
                                    content=SimpleNamespace(parts=[thought, part]))],
        usage_metadata=SimpleNamespace(prompt_token_count=812, candidates_token_count=12, thoughts_token_count=40,
                                       cached_content_token_count=None, total_token_count=864),
        model_version="gemini-x-001")
    r = vg.from_sdk_response(resp)
    assert r["text"] == part.text and r["finish_reason"] == "STOP" and r["model_version"] == "gemini-x-001"
    assert r["usage"] == {"prompt": 812, "output": 12, "thinking": 40, "cached": 0, "total": 864}
    blocked = vg.from_sdk_response(SimpleNamespace(candidates=[], usage_metadata=None,
                                                   prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(
                                                       name="SAFETY"))))
    assert blocked["finish_reason"] == "BLOCKED_SAFETY" and blocked["text"] is None


def test_sdk_request_objects(tmp_path):
    types = pytest.importorskip("google.genai.types")
    img = tmp_path / "q.webp"
    img.write_bytes(b"RIFF0000WEBP")
    r = vg.build_request("m", "CurbRamp", str(img), examples=[(str(img), "Yes")], thinking_level="minimal")
    cfg = vg.to_sdk_config(r["config"], types)
    assert cfg.temperature == 0.0 and cfg.thinking_config.thinking_level.name == "MINIMAL"
    assert cfg.response_mime_type == "application/json"
    contents = vg.to_sdk_contents(r["contents"], types)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[-1].parts[0].inline_data.mime_type == "image/webp"
    assert contents[-1].parts[1].text == r["contents"][-1]["parts"][1]["text"]
    assert vg.to_sdk_config({**r["config"], "thinking_level": None}, types).thinking_config is None


def test_score_frame_never_imports_the_sdk(tmp_path):
    """A stub run never loads google-genai (``post`` is injected), so CI needs no SDK."""
    root, _ = _data_root(tmp_path)
    before = "google.genai" in sys.modules
    vg.main(_args(root, tmp_path, limit=2), post=_Stub())
    assert ("google.genai" in sys.modules) == before
    assert np.isfinite(pd.read_csv(tmp_path / "scores" / "gemini-3.8-flash" / "CurbRamp.csv").p_incorrect).all()


def test_a_meta_with_no_scored_rows_does_not_bind_the_leg(tmp_path):
    """A run that fails before scoring anything leaves a meta and no CSV; the retry with other settings must go
    through (seen on the first pilot: ``minimal`` thinking was refused by the API). With a scored row it binds."""
    path, meta_path = str(tmp_path / "CurbRamp.csv"), str(tmp_path / "CurbRamp.meta.json")
    vg.vb._write_meta(meta_path, {"backend": vg.BACKEND, "binding": {"thinking_level": "minimal"}})
    vg.check_owned(path, meta_path, {"thinking_level": "low"})          # no CSV: allowed
    row = {c: "" for c in vg.SCORE_COLUMNS}
    row["label_uid"] = "x:1"
    vg.append_rows(path, [row])
    with pytest.raises(SystemExit, match="other settings"):
        vg.check_owned(path, meta_path, {"thinking_level": "low"})


# ---- the binding (#105 review) ------------------------------------------------------------------------------------

def _legacy_meta(meta_path):
    """Strip the binding keys a meta written before they were bound lacks (the first real leg's metas)."""
    meta = json.load(open(meta_path))
    for k in vg.LEGACY_UNBOUND_KEYS:
        meta["binding"].pop(k, None)
    vg.vb._write_meta(str(meta_path), meta)


def test_leg_settings_bind_across_label_types(tmp_path):
    """Type A scored under one setting, type B under another, into the same leg: refused, whichever setting differs
    (a check against type B's own meta alone would accept it, since B has none yet)."""
    root, _ = _data_root(tmp_path)
    stub = _Stub()
    vg.main(_args(root, tmp_path, name="g", limit=2, **{"label-types": "CurbRamp", "thinking-level": "low"}),
            post=stub)
    other = tmp_path / "other-cities.txt"
    other.write_text(f"{LISTED}\npub-b\n")
    n = len(stub.requests)
    for kw, key in (({"thinking-level": "high"}, "thinking_level"), ({"model": "gemini-4-argon"}, "model"),
                    ({"cities-file": other}, "cities_sha256")):
        a = {"label-types": "Obstacle", "thinking-level": "low", **kw}
        with pytest.raises(SystemExit, match=key):
            vg.main(_args(root, tmp_path, name="g", limit=2, **a), post=stub)
    assert len(stub.requests) == n
    assert not (tmp_path / "scores" / "g" / "Obstacle.csv").exists()
    vg.main(_args(root, tmp_path, name="g", limit=2, **{"label-types": "Obstacle", "thinking-level": "low"}),
            post=stub)                                   # the same settings go through
    assert (tmp_path / "scores" / "g" / "Obstacle.csv").exists()


def test_query_text_and_fewshot_seed_are_bound(tmp_path, monkeypatch):
    root, _ = _data_root(tmp_path)
    stub = _Stub()
    vg.main(_args(root, tmp_path, limit=4, **{"label-types": "CurbRamp"}), post=stub)
    meta_path = tmp_path / "scores" / "gemini-3.8-flash" / "CurbRamp.meta.json"
    b = json.load(open(meta_path))["binding"]
    assert b["fewshot_seed"] == vb.FEWSHOT_SEED and len(b["query_text_sha256"]) == 64
    n = len(stub.requests)
    with monkeypatch.context() as m:
        m.setitem(vb.DEFINITIONS, "CurbRamp", vb.DEFINITIONS["CurbRamp"] + " Edited.")
        with pytest.raises(SystemExit, match="query_text_sha256"):
            vg.main(_args(root, tmp_path, limit=8, **{"label-types": "CurbRamp"}), post=stub)
    with monkeypatch.context() as m:
        m.setattr(vb, "FEWSHOT_SEED", vb.FEWSHOT_SEED + 1)
        with pytest.raises(SystemExit, match="fewshot_seed"):
            vg.main(_args(root, tmp_path, limit=8, **{"label-types": "CurbRamp"}), post=stub)
    assert len(stub.requests) == n
    assert json.load(open(meta_path))["prompt"] == vb.build_prompt("CurbRamp")       # not overwritten


def test_a_meta_from_before_the_new_binding_keys_still_resumes(tmp_path, capsys):
    """The first real leg's metas carry no ``query_text_sha256`` / ``fewshot_seed``: same arguments resume, with a
    printed note and the unchecked keys recorded in the run; other settings are still refused."""
    root, _ = _data_root(tmp_path)
    stub = _Stub()
    args = dict(limit=4, **{"label-types": "CurbRamp,Obstacle", "thinking-level": "low"})
    vg.main(_args(root, tmp_path, **args), post=stub)
    leg = tmp_path / "scores" / "gemini-3.8-flash"
    for t in ("CurbRamp", "Obstacle"):
        _legacy_meta(leg / f"{t}.meta.json")
    capsys.readouterr()
    vg.main(_args(root, tmp_path, **{**args, "limit": 8}), post=stub)
    out = capsys.readouterr().out
    assert "predates the binding of fewshot_seed, query_text_sha256" in out
    meta = json.load(open(leg / "CurbRamp.meta.json"))
    assert meta["runs"][-1]["binding_unchecked"] == ["fewshot_seed", "query_text_sha256"]
    assert len(pd.read_csv(leg / "CurbRamp.csv")) == 8
    _legacy_meta(leg / "Obstacle.meta.json")
    with pytest.raises(SystemExit, match="thinking_level"):
        vg.main(_args(root, tmp_path, **{**args, "thinking-level": "high"}), post=stub)
