"""Tests for analysis/adjudicator.py (issue #77, #12's scoring step): the gold table, the prompt, scoring against a
stub vLLM server, agreement and the bar, the masked packet, relabel, the split variants, the reliability-weighted
crowd and the report. CPU only; fixtures are synthetic (fake cities city-a / city-b public, city-c private) under
tests/fixtures/adjudicator/."""
import doctest
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from analysis import adjudicator as adj
from analysis import confident_learning as cl
from analysis import vlm_vllm as vv
import fine_tune_classifier as ftc

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "adjudicator")
CONF = os.path.join(FIX, "cityparams.conf")
PUBLIC = os.path.join(FIX, "public_cities.txt")
TEMPLATE_SHA = "c20a02e142e9ce0e4296171f20106028bce5230aa29306f1648db0ad8caabcf3"
OLD = 1760000000  # 2025-10-09: an mtime before any rating


def _fix(name):
    return os.path.join(FIX, name)


def test_doctests():
    assert doctest.testmod(adj).failed == 0


def test_no_torch_or_vllm_import():
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import analysis.adjudicator; "
            "print(sorted(m for m in ('torch', 'transformers', 'vllm') if m in sys.modules))" % adj.REPO_DIR)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
    src = open(adj.__file__).read()
    assert "import torch" not in src and "import vllm" not in src and "import scipy" not in src


# ---------------------------------------------------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------------------------------------------------

def _gold_argv(tmp_path, split_root=None, **kw):
    a = ["gold", "--r1-joined", _fix("r1_joined.csv"), "--r2-key", _fix("r2_key.csv"),
         "--r2-sheets", f"a={_fix('votes_a.csv')}", f"b={_fix('votes_b.csv')}",
         "--r2-gold-shared", _fix("gold_shared.csv"), "--r2-unmasked", _fix("unmasked.csv"), "--cityparams", CONF,
         "--split-root", str(split_root or _fix("splits")), "--out-dir", str(tmp_path / "out")]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a


def _gold(tmp_path, **kw):
    adj.main(_gold_argv(tmp_path, **kw))
    return adj.load_gold(str(tmp_path / "out" / "gold.csv"))


def test_gold_joins_both_key_forms_and_counts(tmp_path, capsys):
    g = _gold(tmp_path, pool=_fix("pool.csv")).set_index("label_uid")
    assert len(g) == 28 and g.index.is_unique and list(adj.load_gold(str(tmp_path / "out" / "gold.csv")).columns) \
        == adj.GOLD_COLUMNS
    assert g.loc["city-c:1", "label_uid_schema"] == "sidewalk_c:1" and g.loc["city-a:101", "round"] == "2"
    # strata recomputed for round 1: voted_disagree splits into margin and light_voted
    assert g.loc["city-a:3", "stratum"] == "margin_incorrect" and g.loc["city-a:4", "stratum"] == "light_voted"
    assert g.loc["city-a:5", "stratum"] == "margin_correct" and g.loc["city-a:3", "group_r1"] == "voted_disagree"
    assert g.expert_source.value_counts().to_dict() == {"r1": 12, "r2_single": 12, "r2_matched": 2, "r2_consensus": 2}
    assert g.loc["city-a:107", "expert"] == "incorrect" and g.loc["city-a:108", "expert"] == "correct"
    assert g.loc["city-b:207", "expert"] == "cant_tell"  # blank gold (no consensus)
    assert g.loc["city-a:107", "expert_source"] == "r2_matched" and g.loc["city-a:108", "expert_source"] == "r2_consensus"
    assert g.loc["city-a:107", "raters"] == "a+b" and g.loc["city-b:201", "raters"] == "b"
    # decided: no cant_tell, no unmasked
    assert not g.loc["city-a:7", "decided"] and not g.loc["city-b:204", "decided"]
    assert g.loc["city-a:106", "unmasked"] and not g.loc["city-a:106", "decided"]
    assert int(g.decided.sum()) == 28 - 4  # 2 cant_tell in r1/r2 sheets, 1 blank gold, 1 unmasked
    # the pool refresh: a later vote moves the crowd majority
    assert g.loc["city-a:101", "crowd_class"] == "correct"
    s = json.load(open(tmp_path / "out" / "gold_summary.json"))
    assert s["pool_rows_found"] == 17 and s["n_unmasked"] == 1  # the 16 round-2 items and city-a:1
    assert open(tmp_path / "out" / "gold.sha256").read().split()[0] == s["gold_sha256"]
    fc = pd.read_csv(tmp_path / "out" / "gold_for_cl.csv", dtype=str)
    assert list(fc.columns) == ["label_uid", "verdict"] and len(fc) == int(g.decided.sum())
    assert set(fc.verdict) == {"correct", "incorrect"} and cl.adjudicate_gold(fc).label_uid.is_unique
    assert "| all | all |" in capsys.readouterr().out


def test_gold_refuses_key_form_mismatch(tmp_path):
    key = pd.read_csv(_fix("r2_key.csv"), dtype=str, keep_default_na=False)
    adj.check_key_forms(key, adj.schema_city_map(open(CONF).read()))
    key.loc[0, "label_uid_city"] = "city-b:" + key.loc[0, "label_id"]
    with pytest.raises(ValueError, match="key form mismatch"):
        adj.check_key_forms(key)
    bad = tmp_path / "key.csv"
    key.to_csv(bad, index=False)
    argv = _gold_argv(tmp_path)
    argv[argv.index("--r2-key") + 1] = str(bad)
    with pytest.raises(SystemExit, match="key form mismatch"):
        adj.main(argv)


def test_gold_refuses_unknown_schema(tmp_path):
    j = pd.read_csv(_fix("r1_joined.csv"))
    j.loc[0, ["label_uid", "schema"]] = ["sidewalk_zz:1", "sidewalk_zz"]
    p = tmp_path / "r1.csv"
    j.to_csv(p, index=False)
    argv = _gold_argv(tmp_path)
    argv[argv.index("--r1-joined") + 1] = str(p)
    with pytest.raises(SystemExit, match="sidewalk_zz"):
        adj.main(argv)


def test_gold_refuses_an_item_in_two_rounds():
    sc = adj.schema_city_map(open(CONF).read())
    r1 = adj.round1_rows(pd.read_csv(_fix("r1_joined.csv")), sc)
    with pytest.raises(ValueError, match="more than one round"):
        adj.combine_rounds(r1, r1.head(1))


def test_gold_train_row_and_cl_flags(tmp_path):
    g = _gold(tmp_path).set_index("label_uid")
    assert g.loc["city-a:1", "train_row"] and g.loc["city-a:1", "round3_split"] == "train"
    assert g.loc["city-a:3", "round3_split"] == "val" and not g.loc["city-a:3", "train_row"]
    assert g.loc["city-a:101", "class_source_r3"] == "expert" and g.loc["city-a:101", "cl_flagged"]
    assert not g.loc["city-a:1", "cl_flagged"]  # noisy_targets lists it, flagged 0
    assert g.loc["city-a:2", "crop_path_r3"] == "correct/city-a_2.webp" and g.loc["city-a:2", "crop_status_r3"] == "ok"
    assert g.loc["city-a:5", "round3_split"] == ""  # not in round 3
    assert int(g.train_row.sum()) == 6


def test_gold_summary_has_no_ids(tmp_path):
    _gold(tmp_path)
    text = open(tmp_path / "out" / "gold_summary.json").read()
    for bad in ("city-a:", "city-c", "sidewalk_c", "r2a", "r1000"):
        assert bad not in text
    s = json.loads(text)
    assert sum(c["n"] for c in s["cells"]) == 28


def test_gold_refuses_output_inside_a_git_checkout(tmp_path):
    argv = _gold_argv(tmp_path)
    argv[argv.index("--out-dir") + 1] = os.path.join(adj.REPO_DIR, "scratch-gold")
    with pytest.raises(SystemExit, match="git checkout"):
        adj.main(argv)


# ---------------------------------------------------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------------------------------------------------

def test_crops_lists_labels_staging_and_missing_geometry(tmp_path, capsys):
    _gold(tmp_path)
    out = tmp_path / "crops"
    adj.main(["crops", "--gold", str(tmp_path / "out" / "gold.csv"), "--rawlabels", _fix("rawlabels"),
              "--out-dir", str(out)])
    labels = pd.read_csv(out / "labels.csv", dtype=str, keep_default_na=False)
    from analysis import ai_labels_eval as ae
    assert list(labels.columns) == ae.LABEL_COLUMNS
    existing = pd.read_csv(out / "existing_crops.csv")
    assert set(existing.label_uid) & set(labels.label_uid) == set()
    assert "city-a:3" in set(labels.label_uid) or "city-a:3" in set(existing.label_uid)
    assert set(labels.label_class) == {""}  # the crop leg never sees the expert verdict
    assert labels.set_index("label_uid").loc["city-a:4", "pano_x"] == "1004"
    missing = pd.read_csv(out / "geometry_missing.csv")
    assert "city-c:1" in set(missing.label_uid)  # private, not in the cache
    assert "sidewalk_c:1" in open(out / "label_points_request.txt").read()
    staged = open(out / "stage_files.txt").read().split()
    assert all(p.count("/") == 2 and p.endswith(".jpg") for p in staged)
    # with the label-points pull, the private items become scorable
    lp = tmp_path / "lp.csv"
    pd.DataFrame({"schema": "sidewalk_c", "label_id": [1, 2], "pano_id": ["C1", "C2"], "pano_source": "gsv",
                  "author_is_ai": ["t", "t"], "pano_x": 1, "pano_y": 2, "pano_width": 4, "pano_height": 2,
                  "camera_pitch": 0, "camera_roll": ""}).to_csv(lp, index=False)
    adj.main(["crops", "--gold", str(tmp_path / "out" / "gold.csv"), "--rawlabels", _fix("rawlabels"),
              "--label-points", str(lp), "--out-dir", str(tmp_path / "crops2")])
    l2 = pd.read_csv(tmp_path / "crops2" / "labels.csv", dtype=str, keep_default_na=False).set_index("label_uid")
    assert l2.loc["city-c:1", "author"] == "ai"  # from the pull
    assert l2.loc["city-c:2", "author"] == "human"  # the cache file wins where it has the label


def test_stage_list_prints_panos_and_gb(tmp_path, capsys):
    adj.main(["stage-list", "--pool", _fix("pool.csv"), "--types", "CurbRamp", "--cityparams", CONF,
              "--out-dir", str(tmp_path / "pool")])
    out = capsys.readouterr().out
    assert "AI Disagrees of ['CurbRamp']" in out and "GB to stage" in out
    items = pd.read_csv(tmp_path / "pool" / "pool_items.csv")
    assert items.label_uid.str.startswith("city-").all() and items.label_uid.is_unique
    # the pool item list feeds the same crops stage
    adj.main(["crops", "--gold", str(tmp_path / "pool" / "pool_items.csv"), "--rawlabels", _fix("rawlabels"),
              "--out-dir", str(tmp_path / "pool_crops")])
    labels = pd.read_csv(tmp_path / "pool_crops" / "labels.csv")
    assert "city-a:103" in set(labels.label_uid)


# ---------------------------------------------------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------------------------------------------------

def test_prompt_renders_definition_rule_severity_and_question():
    p = adj.build_adjudicator_prompt("CurbRamp", 4.0)
    assert p.startswith(adj.vb.DEFINITIONS["CurbRamp"]) and adj.RULE["CurbRamp"] in p
    assert "The labelled point is at the centre of the image." in p
    assert "rated its severity 4 on a 1 (minor) to 5 (severe) scale" in p
    assert p.endswith("Is this a correct curb ramp label? Answer yes or no.")
    assert "gave no severity" in adj.build_adjudicator_prompt("CurbRamp", "")
    assert "Tags" not in adj.build_adjudicator_prompt("CurbRamp", 1, "steep")


def test_prompt_hides_ai_and_crowd():
    for t in adj.LABEL_TYPES:
        for show in (False, True):
            p = adj.build_adjudicator_prompt(t, 2, "narrow;pole", show)
            words = set(p.replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").split())
            assert not words & {"AI", "votes", "vote", "crowd", "validators", "production", "Disagree", "Agree"}
            assert "crowd" not in p.lower() and "vote" not in p.lower()


def test_prompt_template_hash_is_stable_and_changes_with_rule(monkeypatch):
    assert adj.prompt_template_hash() == TEMPLATE_SHA
    assert adj.prompt_template_hash(True) != TEMPLATE_SHA
    monkeypatch.setitem(adj.RULE, "Obstacle", adj.RULE["Obstacle"] + " Changed.")
    assert adj.prompt_template_hash() != TEMPLATE_SHA
    # the rendered severity does not change the template hash, only the prompt hash
    monkeypatch.undo()
    a, b = adj.build_adjudicator_prompt("Obstacle", 1), adj.build_adjudicator_prompt("Obstacle", 5)
    assert adj.prompt_hash(a) != adj.prompt_hash(b) and adj.prompt_template_hash() == TEMPLATE_SHA


def test_show_tags_variant(capsys):
    p = adj.build_adjudicator_prompt("Obstacle", 2, '["narrow","pole"]', show_tags=True)
    assert "Tags: narrow, pole." in p
    assert "Tags: none." in adj.build_adjudicator_prompt("Obstacle", 2, "", show_tags=True)
    adj.main(["prompt", "--show-tags"])
    out = capsys.readouterr().out
    assert out.count("## ") == 5 and adj.prompt_template_hash(True) in out


def test_reason_request_shape():
    b = adj.build_reason_request("m", "data:,", "Q?", "Yes.", template_kwargs={"enable_thinking": False})
    assert [m["role"] for m in b["messages"]] == ["user", "assistant", "user"]
    assert b["messages"][1]["content"] == "Yes." and b["messages"][2]["content"] == adj.REASON_QUESTION
    assert b["max_tokens"] == 60 and b["temperature"] == 0.0 and "logprobs" not in b and "top_logprobs" not in b
    assert b["chat_template_kwargs"] == {"enable_thinking": False} and "continue_final_message" not in b
    g = adj.build_reason_request("m", "data:,", "Q?", "No.", prefill="<think></think>")
    assert g["messages"][3] == {"role": "assistant", "content": "<think></think>"}
    assert g["continue_final_message"] and not g["add_generation_prompt"]


# ---------------------------------------------------------------------------------------------------------------------
# score (stub vLLM server)
# ---------------------------------------------------------------------------------------------------------------------

def _resp(pairs, prompt_tokens=700):
    top = [{"token": t, "logprob": math.log(p)} for t, p in pairs]
    return {"choices": [{"message": {"content": pairs[0][0]},
                         "logprobs": {"content": [{"token": pairs[0][0], "logprob": top[0]["logprob"],
                                                   "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1}}


class _Stub:
    """Verdict requests: p(no) 0.8 for obstacles, 0.3 otherwise; reason requests: a fixed sentence."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, body):
        self.calls.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 3, "token_strs": ["<user>", "Q", "<assistant>"]}
        if "logprobs" not in body:
            return {"choices": [{"message": {"content": "  The ramp is\nthere. "}}],
                    "usage": {"completion_tokens": 7}}
        q = body["messages"][0]["content"][1]["text"]
        p_no = 0.8 if "obstacle label" in q else 0.3
        return _resp([("No", p_no * 0.99), ("Yes", (1 - p_no) * 0.99)])


def _split_root(tmp_path):
    """The fixture splits with stub crop files beside them."""
    root = tmp_path / "data"
    shutil.copytree(FIX + "/splits", root)
    for t in ("CurbRamp", "Obstacle"):
        s = cl.read_split(str(root / t / "split.csv"))
        for p in s.crop_path:
            os.makedirs(root / t / os.path.dirname(p), exist_ok=True)
            (root / t / p).write_bytes(b"RIFF0000WEBP")
    return root


def _crops_root(tmp_path, uids):
    c = tmp_path / "crops"
    (c / "crops").mkdir(parents=True)
    rows = []
    for u in uids:
        p = f"crops/{u.replace(':', '_')}.webp"
        (c / p).write_bytes(b"RIFF0000WEBP")
        rows.append({"label_uid": u, "crop_path": p, "status": "ok"})
    rows.append({"label_uid": "city-a:6", "crop_path": "crops/none.webp", "status": "no_pano"})
    pd.DataFrame(rows).to_csv(c / "manifest.csv", index=False)
    return c


@pytest.fixture
def stub(monkeypatch):
    s = _Stub()
    orig = adj.AdjudicatorClient.__init__

    def init(self, *a, **k):
        orig(self, *a, **{**k, "post": s})
    monkeypatch.setattr(adj.AdjudicatorClient, "__init__", init)
    monkeypatch.setattr(vv, "_server_version", lambda server: "0.0-test")
    return s


def _score_argv(gold, root, crops, out, **kw):
    a = ["score", "--model", "stub", "--repo", "org/stub", "--gold", str(gold), "--split-root", str(root),
         "--crops-root", str(crops), "--out-dir", str(out), "--gpu", "1xTEST", "--concurrency", "2",
         "--hf-home", str(out)]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return a


def _scored(tmp_path, stub, **kw):
    gold = tmp_path / "out" / "gold.csv"
    if not gold.exists():
        _gold(tmp_path, split_root=_split_root(tmp_path))
    root = tmp_path / "data"
    crops = _crops_root(tmp_path, ["city-a:4", "city-a:5", "city-b:201"]) if not (tmp_path / "crops").exists() \
        else tmp_path / "crops"
    out = tmp_path / "scores"
    adj.main(_score_argv(gold, root, crops, out, **kw))
    return gold, root, crops, out


def test_score_writes_adj_columns_verdict_reason_and_resumes(tmp_path, stub):
    gold, root, crops, out = _scored(tmp_path, stub, prefill="<think></think>")
    s = pd.read_csv(out / "stub" / "gold.csv", keep_default_na=False)
    assert list(s.columns) == adj.ADJ_COLUMNS and s.label_uid.is_unique
    g = adj.load_gold(str(gold)).set_index("label_uid")
    assert len(s) == 3 + int(((g.crop_status_r3 == "ok") & ~g.index.isin(["city-a:4", "city-a:5", "city-b:201"])).sum())
    s = s.set_index("label_uid")
    assert s.loc["city-a:2", "verdict"] == "incorrect" and s.loc["city-a:1", "verdict"] == "correct"
    assert s.loc["city-a:1", "reason"] == "The ramp is there." and s.loc["city-a:1", "reason_tokens"] == 7
    assert s.loc["city-a:1", "prompt_variant"] == "no-tags" and s.loc["city-a:1", "variant"] == adj.PROMPT_VERSION
    q = adj.build_adjudicator_prompt("CurbRamp", g.loc["city-a:1", "severity"], g.loc["city-a:1", "tags"])
    assert s.loc["city-a:1", "prompt_sha256"] == adj.prompt_hash(q)
    meta = json.load(open(out / "stub" / "gold.meta.json"))
    assert meta["backend"] == adj.ADJ_BACKEND and meta["prompt_template_sha256"] == TEMPLATE_SHA
    assert meta["gold_sha256"] and meta["rendered_prompt_tail"]["count"] == 3 and meta["reason"]
    assert meta["request"]["prefill"] == "<think></think>" and meta["n_no_image"] == 28 - len(s)
    assert "city-a" not in json.dumps(meta) and "label_uid" not in json.dumps(meta)
    # the reason request carries the prefill turn too
    reason_bodies = [b for u, b in stub.calls if u.endswith("/chat/completions") and "logprobs" not in b]
    assert reason_bodies and all(b["messages"][-1]["content"] == "<think></think>" for b in reason_bodies)
    n_calls = len(stub.calls)
    adj.main(_score_argv(gold, root, crops, out, prefill="<think></think>"))
    assert len(stub.calls) == n_calls  # nothing left to score
    # a truncated CSV is completed without duplicates or repeated requests for done rows
    full = pd.read_csv(out / "stub" / "gold.csv", keep_default_na=False)
    full.head(1).to_csv(out / "stub" / "gold.csv", index=False)
    adj.main(_score_argv(gold, root, crops, out))
    again = pd.read_csv(out / "stub" / "gold.csv", keep_default_na=False)
    assert len(again) == len(full) and again.label_uid.is_unique
    new = [b for u, b in stub.calls[n_calls:] if u.endswith("/chat/completions") and "logprobs" in b]
    assert len(new) == len(full) - 1


def test_score_no_reason_flag(tmp_path, stub):
    _, _, _, out = _scored(tmp_path, stub, no_reason=True)
    s = pd.read_csv(out / "stub" / "gold.csv", keep_default_na=False)
    assert set(s.reason) == {""}
    assert all("logprobs" in b for u, b in stub.calls if u.endswith("/chat/completions"))
    assert not json.load(open(out / "stub" / "gold.meta.json"))["reason"]


def test_score_refuses_foreign_files(tmp_path, stub):
    (tmp_path / "scores" / "stub").mkdir(parents=True)
    pd.DataFrame({"label_uid": ["x"], "p_incorrect": [0.5]}).to_csv(tmp_path / "scores" / "stub" / "gold.csv",
                                                                      index=False)
    with pytest.raises(SystemExit, match="not written by adjudicator.py"):
        _scored(tmp_path, stub)
    with pytest.raises(SystemExit, match="git checkout"):
        adj.main(_score_argv(tmp_path / "out" / "gold.csv", tmp_path / "data", tmp_path / "crops",
                             os.path.join(adj.REPO_DIR, "scratch-scores")))


def test_score_uses_round3_crop_when_no_staged_crop(tmp_path):
    root = _split_root(tmp_path)
    crops = _crops_root(tmp_path, ["city-a:1"])
    g = _gold(tmp_path, split_root=root)
    r = adj.resolve_images(g, str(crops), str(root)).set_index("label_uid")
    assert r.loc["city-a:1", "image"] == os.path.join(str(crops), "crops/city-a_1.webp")  # staged wins
    assert r.loc["city-a:2", "image"] == os.path.join(str(root), "Obstacle", "correct/city-a_2.webp")
    assert r.loc["city-a:5", "image"] == ""  # neither
    assert r.loc["city-a:6", "image"] == ""  # staged row not ok, no round-3 crop


def test_limit_per_type():
    d = pd.DataFrame({"label_type": ["A"] * 3 + ["B"] * 2, "label_uid": ["a3", "a1", "a2", "b2", "b1"]})
    assert list(adj.limit_per_type(d, 1).label_uid) == ["a1", "b1"] and len(adj.limit_per_type(d, 0)) == 5


# ---------------------------------------------------------------------------------------------------------------------
# agree
# ---------------------------------------------------------------------------------------------------------------------

def _m(rows, model="M"):
    base = {"label_type": "CurbRamp", "stratum": "lone", "ai_vote": "Disagree", "crowd_class": "", "severity": "2",
            "tags": "", "city_id": "city-a", "p_incorrect": 0.5, "model": model}
    df = pd.DataFrame([{**base, "label_uid": f"city-a:{i}", "pano_id": f"p{i}", **r} for i, r in enumerate(rows)])
    df["verdict"] = np.where(df.p_incorrect > 0.5, "incorrect", "correct")
    return df


def test_agreement_rows_by_hand():
    rows = ([{"expert": "incorrect", "p_incorrect": 0.9, "crowd_class": "incorrect"}] * 6
            + [{"expert": "correct", "p_incorrect": 0.2, "crowd_class": "incorrect"}] * 2
            + [{"expert": "correct", "p_incorrect": 0.7}] * 2)
    a = adj.agreement_rows(_m(rows), n_boot=0).set_index(["model", "label_type"])
    r = a.loc[("M", "all")]
    assert r.n == 10 and r.vlm_agree == pytest.approx(0.8)
    lo, hi = adj.rl.wilson_interval(0.8, 10)
    assert (r.lo, r.hi) == pytest.approx((float(lo), float(hi)))
    assert r.crowd_n == 8 and r.crowd_agree == pytest.approx(6 / 8)
    assert r.ai_disagree_n == 10 and r.ai_confirmed == pytest.approx(0.6)
    assert r.auc_vs_expert == 1.0  # every incorrect item scores 0.9, above every correct one (0.2, 0.7)
    assert ("M", "CurbRamp") in a.index and ("M", "Obstacle") not in a.index


def test_paired_delta_and_mcnemar_by_hand():
    # 10 paired items: model right & crowd wrong on 4, crowd right & model wrong on 1, both right on 5
    rows = ([{"expert": "incorrect", "p_incorrect": 0.9, "crowd_class": "correct"}] * 4
            + [{"expert": "incorrect", "p_incorrect": 0.1, "crowd_class": "incorrect"}]
            + [{"expert": "correct", "p_incorrect": 0.1, "crowd_class": "correct"}] * 5)
    p = adj.paired_rows(_m(rows), n_boot=500, seed=1).set_index("label_type").loc["all"]
    assert p.n_pair == 10 and p.vlm_agree == pytest.approx(0.9) and p.crowd_agree == pytest.approx(0.6)
    assert p.delta == pytest.approx(0.3) and p.b_vlm_only == 4 and p.b_crowd_only == 1
    assert p.mcnemar_p == pytest.approx(2 * (1 + 5) / 32)
    assert p.delta_lo <= 0.3 <= p.delta_hi
    assert adj.mcnemar_exact(3, 3) == 1.0


def test_shuffled_verdicts_drop_agreement_to_chance():
    rng = np.random.default_rng(7)
    n = 400
    expert = rng.choice(["correct", "incorrect"], n)
    crowd = np.where(rng.random(n) < 0.75, expert, np.where(expert == "correct", "incorrect", "correct"))
    m = _m([{"expert": e, "crowd_class": c, "p_incorrect": 0.5} for e, c in zip(expert, crowd)])
    m["verdict"] = rng.permutation(np.where(rng.random(n) < 0.5, "correct", "incorrect"))
    a = adj.agreement_rows(m, n_boot=0).set_index("label_type").loc["all"]
    p = adj.paired_rows(m, n_boot=300, seed=0).set_index("label_type").loc["all"]
    assert abs(a.vlm_agree - 0.5) < 0.08 and p.delta_lo <= 0
    assert not adj.bar_decision(p.to_dict(), a.to_dict())["clears"]


def test_strata_rows():
    rows = ([{"stratum": "lone", "expert": "incorrect", "p_incorrect": 0.9}] * 3
            + [{"stratum": "margin_correct", "expert": "correct", "p_incorrect": 0.9, "crowd_class": "correct"}] * 2)
    s = adj.stratum_rows(_m(rows)).set_index("stratum")
    assert s.loc["lone", "vlm_agree"] == 1.0 and s.loc["lone", "crowd_n"] == 0
    assert s.loc["margin_correct", "vlm_agree"] == 0.0 and s.loc["margin_correct", "crowd_agree"] == 1.0
    assert s.loc["all", "n"] == 5


def test_slice_tables_use_slices_py_with_training_counts():
    rows = ([{"expert": "incorrect", "p_incorrect": 0.9, "severity": "1", "tags": "steep"}] * 6
            + [{"expert": "correct", "p_incorrect": 0.2, "severity": "1", "tags": "steep;narrow"}] * 6)
    train = pd.DataFrame({"label_type": ["CurbRamp"] * 3 + ["Obstacle"], "severity": ["1", "1", "3", "1"],
                          "tags": ['["steep"]', "[]", "[]", '["steep"]']})
    t = adj.slice_tables(_m(rows), train, min_n=10, n_boot=0)
    sev = t["severity"].set_index("slice")
    assert sev.loc["1", "n"] == 12 and sev.loc["1", "vlm_agree"] == 1.0 and sev.loc["1", "train_n"] == 2
    tag = t["tag"].set_index("slice")
    assert tag.loc["steep", "train_n"] == 1 and tag.loc["narrow", "n"] == 6 and pd.isna(tag.loc["narrow", "vlm_agree"])
    assert list(t["tag"].columns[:2]) == ["model", "label_type"]


@pytest.mark.parametrize("paired,all_row,clears,why", [
    ({"n_pair": 30, "delta_lo": 0.02, "crowd_agree": 0.70}, {"vlm_agree": 0.80}, True, "n_pair 30"),
    ({"n_pair": 29, "delta_lo": 0.20, "crowd_agree": 0.70}, {"vlm_agree": 0.90}, False, "n_pair 29 < 30"),
    ({"n_pair": 50, "delta_lo": 0.0, "crowd_agree": 0.70}, {"vlm_agree": 0.90}, False, "not above 0"),
    ({"n_pair": 50, "delta_lo": -0.1, "crowd_agree": 0.70}, {"vlm_agree": 0.90}, False, "not above 0"),
    ({"n_pair": 50, "delta_lo": 0.05, "crowd_agree": 0.70}, {"vlm_agree": 0.69}, False, "< crowd"),
    ({"n_pair": 50, "delta_lo": 0.05, "crowd_agree": 0.70}, {"vlm_agree": 0.70}, True, ">= crowd"),
])
def test_bar_decision(paired, all_row, clears, why):
    d = adj.bar_decision(paired, all_row)
    assert d["clears"] is clears and why in d["reason"]


def test_agree_cli_writes_aggregates_and_bar(tmp_path, stub):
    gold, root, crops, out = _scored(tmp_path, stub, no_reason=True)
    adj.main(["agree", "--gold", str(gold), "--scores", str(out), "--models", "stub", "--split-root", str(root),
              "--out-dir", str(tmp_path / "agree"), "--n-boot", "50"])
    for f in ("agreement.csv", "paired.csv", "strata.csv", "slices_severity.csv", "slices_tag.csv", "bar.json",
              "agree.md"):
        assert (tmp_path / "agree" / f).exists()
        text = (tmp_path / "agree" / f).read_text()
        assert "city-a:" not in text and "city-c" not in text
    bar = json.load(open(tmp_path / "agree" / "bar.json"))
    assert set(bar["decisions"]["stub"]) <= {"all", "CurbRamp", "Obstacle"} and bar["min_n"] == 30
    assert not any(d["clears"] for d in bar["decisions"]["stub"].values())  # n < 30 in the fixture
    md = (tmp_path / "agree" / "agree.md").read_text()
    assert "reference: crowd majority, round 1" in md and "74.5% [64.8, 82.2]" in md


# ---------------------------------------------------------------------------------------------------------------------
# disagreements packet
# ---------------------------------------------------------------------------------------------------------------------

def test_disagreements_packet_masked(tmp_path, stub):
    gold, root, crops, out = _scored(tmp_path, stub)
    pk, key = tmp_path / "packet", tmp_path / "key"
    adj.main(["disagreements", "--gold", str(gold), "--scores", str(out), "--model", "stub", "--crops-root",
              str(crops), "--split-root", str(root), "--out-dir", str(pk), "--key-dir", str(key), "--seed", "s"])
    k = pd.read_csv(key / "key.csv", keep_default_na=False)
    assert list(k.columns) == ["rid", "label_uid", "expert", "crowd_class", "ai_vote"]
    assert 0 < len(k) <= 30 and k.label_uid.is_unique
    g = adj.load_gold(str(gold)).set_index("label_uid")
    assert (g.loc[k.label_uid, "decided"]).all()
    sheet = (pk / "sheet.md").read_text()
    for bad in ("expert", "crowd", "Disagree", "AI", "city-a"):
        assert bad not in sheet
    assert all(r in sheet for r in k.rid)
    assert len(os.listdir(pk / "items")) == len(k)
    man = json.load(open(pk / "packet_manifest.json"))
    assert man["key_sha256"] == adj.rl.sha256_file(str(key / "key.csv")) and max(man["per_type"].values()) <= 6
    m = pd.DataFrame({"label_uid": [f"x:{i}" for i in range(20)], "label_type": "A", "verdict": "correct",
                      "expert": "incorrect"})
    assert len(adj.pick_disagreements(m, "s")) == 6
    with pytest.raises(SystemExit, match="git checkout"):
        adj.main(["disagreements", "--gold", str(gold), "--scores", str(out), "--model", "stub", "--out-dir", str(pk),
                  "--key-dir", os.path.join(adj.REPO_DIR, "scratch-key"), "--seed", "s"])


# ---------------------------------------------------------------------------------------------------------------------
# relabel and variants
# ---------------------------------------------------------------------------------------------------------------------

def test_contested_rows_modes():
    split = cl.read_split(_fix("splits/CurbRamp/split.csv"))
    noisy = pd.read_csv(_fix("splits/CurbRamp/cl/noisy_targets.csv"), dtype=str)
    ai = {"city-a:1", "city-a:900", "city-a:3", "city-b:902"}  # val / test_heldout rows are never contested
    both = adj.contested_rows(split, ai, noisy).set_index("label_uid")
    assert set(both.index) == {"city-a:1", "city-a:900", "city-a:101"}
    assert both.loc["city-a:900", "contested_by"] == "both" and both.loc["city-a:101", "contested_by"] == "cl_flagged"
    assert set(adj.contested_rows(split, ai, noisy, "ai-disagree").label_uid) == {"city-a:1", "city-a:900"}
    assert set(adj.contested_rows(split, ai, noisy, "cl-flagged").label_uid) == {"city-a:900", "city-a:101"}
    with pytest.raises(ValueError):
        adj.contested_rows(split, ai, noisy, "other")


def test_relabel_refuses_uncleared_pair_without_force(tmp_path, stub, capsys):
    root = _split_root(tmp_path)
    bar = tmp_path / "bar.json"
    json.dump({"decisions": {"stub": {"CurbRamp": {"clears": False, "reason": "n_pair 12 < 30"},
                                      "Obstacle": {"clears": True, "reason": "ok"}}}}, open(bar, "w"))
    argv = ["relabel", "--model", "stub", "--repo", "org/stub", "--gpu", "1xTEST", "--concurrency", "2",
            "--hf-home", str(tmp_path), "--split-root", str(root), "--bar", str(bar), "--pool", _fix("pool.csv"),
            "--cityparams", CONF, "--out-dir", str(tmp_path / "relabel")]
    with pytest.raises(SystemExit, match="does not clear the bar for CurbRamp"):
        adj.main(argv + ["--types", "CurbRamp"])
    assert not any(u.endswith("/chat/completions") for u, _ in stub.calls)
    adj.main(argv + ["--types", "CurbRamp", "--force"])
    meta = json.load(open(tmp_path / "relabel" / "stub" / "train_CurbRamp.meta.json"))
    assert meta["forced"] and not meta["bar_cleared"] and not meta["reason"] and meta["pool_sha256"]
    s = pd.read_csv(tmp_path / "relabel" / "stub" / "train_CurbRamp.csv", keep_default_na=False)
    con = pd.read_csv(tmp_path / "relabel" / "stub" / "contested_CurbRamp.csv")
    # pool AI Disagrees among the CurbRamp train rows, plus city-a:101 (an AI Disagree and flagged)
    assert set(con.label_uid) == {"city-a:1", "city-a:900", "city-a:103", "city-a:101", "city-b:201"} \
        == set(s.label_uid)
    assert set(s.reason) == {""}  # no reasons on relabel runs by default
    adj.main(argv + ["--types", "Obstacle"])
    assert json.load(open(tmp_path / "relabel" / "stub" / "train_Obstacle.meta.json"))["bar_cleared"]


def _base_split():
    return cl.read_split(_fix("splits/CurbRamp/split.csv"))


def test_adj_replace_changes_only_contested_train_rows():
    base = _base_split()
    contested = {"city-a:1", "city-a:101", "city-a:900", "city-a:3"}
    verdicts = {"city-a:1": "correct", "city-a:101": "correct", "city-a:900": "correct", "city-a:3": "correct"}
    v = adj.adj_replace(base, verdicts, contested).set_index("label_uid")
    b = base.set_index("label_uid")
    assert v.loc["city-a:1", "label_class"] == "correct" and v.loc["city-a:1", "class_source"] == "adjudicator"
    assert v.loc["city-a:101", "label_class"] == "incorrect"  # expert-sourced: spared
    assert v.loc["city-a:900", "class_source"] == "margin"  # unchanged target keeps its source
    assert v.loc["city-a:3", "label_class"] == "incorrect"  # val row untouched
    changed = (v.label_class != b.label_class)
    assert set(v.index[v.class_source == "adjudicator"]) == set(v.index[changed]) == {"city-a:1"}
    ev = ~b.split.isin(["train"])
    pd.testing.assert_frame_equal(v[ev], b[ev])
    v2 = adj.adj_replace(base, verdicts, contested, spare_expert=False).set_index("label_uid")
    assert v2.loc["city-a:101", "label_class"] == "correct"
    # the written file round-trips through read_split with eval rows byte-identical
    import tempfile
    d = tempfile.mkdtemp()
    cl.write_split(v.reset_index(), os.path.join(d, "split.csv"))
    back = cl.read_split(os.path.join(d, "split.csv"))
    cl.check_eval_rows_unchanged(base, back)


def test_adj_soft_blend_arithmetic():
    assert adj.soft_blend(0, 2, 1.0, 2.0) == 0.5
    assert adj.soft_blend(3, 1, 0.25, 0.0) == ftc.smoothed_rate(3, 1)
    assert adj.soft_blend(1, 1, 0.0, 2.0) == pytest.approx(2 / 6)
    base = _base_split()
    v = adj.adj_soft(base, {"city-a:1": 1.0, "city-a:900": 0.0, "city-a:101": 1.0}, {"city-a:1", "city-a:900",
                                                                                     "city-a:101"}, w=2.0)
    s = v.set_index("label_uid").soft_target
    assert float(s["city-a:1"]) == pytest.approx((0 + 1 + 2) / (0 + 2 + 2 + 2))  # human 0 agree, 2 disagree
    assert float(s["city-a:900"]) == pytest.approx((3 + 1 + 0) / (3 + 0 + 2 + 2))
    assert float(s["city-a:101"]) == pytest.approx(ftc.smoothed_rate(0, 3))  # expert: spared
    assert float(s["city-b:201"]) == pytest.approx(ftc.smoothed_rate(3, 0))  # not contested
    assert s["city-a:3"] == "" and s["city-b:902"] == ""  # non-train rows blank
    assert list(v.label_class) == list(base.label_class)
    w0 = adj.adj_soft(base, {"city-a:1": 1.0}, {"city-a:1"}, w=0.0).set_index("label_uid").soft_target
    assert float(w0["city-a:1"]) == pytest.approx(ftc.smoothed_rate(0, 2))


def test_soft_target_hook_in_fine_tune():
    d = pd.DataFrame({"label_uid": ["a:1", "a:2"], "human_agree": [3, 0], "human_disagree": [0, 2],
                      "soft_target": ["0.5", "0.125"]}, index=[7, 9])
    out = ftc.soft_targets(d, "soft")
    assert out.tolist() == [0.5, 0.125] and list(out.index) == [7, 9]
    assert ftc.soft_targets(d.drop(columns="soft_target"), "soft").tolist() == [0.8, 0.25]
    with pytest.raises(ValueError, match="soft_target"):
        ftc.soft_targets(d.assign(soft_target=["0.5", ""]), "soft")
    with pytest.raises(ValueError, match="soft_target"):
        ftc.soft_targets(d.assign(soft_target=["1.5", "0.2"]), "soft")
    w = pd.DataFrame({"label_uid": ["a:1", "a:2"], "w_agree": [2.0, 0.0], "w_disagree": [0.0, 1.5]})
    assert ftc.soft_targets(d, "weighted", w).tolist() == [0.75, 0.2857142857142857]  # weighted ignores the column


def test_variants_json_has_no_ids(tmp_path, stub):
    root = _split_root(tmp_path)
    relabel = tmp_path / "relabel" / "stub"
    relabel.mkdir(parents=True)
    rows = [{"label_uid": u, "model": "stub", "p_incorrect": p, "verdict": "incorrect" if p > 0.5 else "correct"}
            for u, p in (("city-a:1", 0.2), ("city-a:900", 0.9), ("city-a:101", 0.1))]
    pd.DataFrame(rows).reindex(columns=adj.ADJ_COLUMNS).to_csv(relabel / "train_CurbRamp.csv", index=False)
    pd.DataFrame({"label_uid": ["city-a:1", "city-a:900", "city-a:101"],
                  "contested_by": ["ai_disagree", "both", "cl_flagged"]}).to_csv(relabel / "contested_CurbRamp.csv",
                                                                               index=False)
    json.dump({"backend": adj.ADJ_BACKEND, "revision": "abc", "prompt_template_sha256": TEMPLATE_SHA},
              open(relabel / "train_CurbRamp.meta.json", "w"))
    adj.main(["variants", "--split-root", str(root), "--relabel", str(tmp_path / "relabel"), "--model", "stub",
              "--types", "CurbRamp"])
    for name in adj.VARIANTS:
        text = (root / "CurbRamp" / "cl" / name / "variant.json").read_text()
        assert "city-a" not in text and "label_uid" not in text
        j = json.loads(text)
        assert j["rows_contested"] == 3 and j["rows_scored"] == 3 and j["model_revision"] == "abc"
        v = cl.read_split(str(root / "CurbRamp" / "cl" / name / "split.csv"))
        cl.check_eval_rows_unchanged(_base_split(), v[_base_split().columns])
    rep = json.loads((root / "CurbRamp" / "cl" / "adj-replace" / "variant.json").read_text())
    assert rep["changed"] == 2  # city-a:1 -> correct, city-a:900 -> incorrect; city-a:101 is expert-sourced
    gold = _write_gold(tmp_path, root)
    adj.main(["variants", "--split-root", str(root), "--relabel", str(tmp_path / "relabel"), "--model", "stub",
              "--types", "CurbRamp", "--contested", "ai-disagree", "--drop-gold", "--gold", gold])
    rep = json.loads((root / "CurbRamp" / "cl" / "adj-replace" / "variant.json").read_text())
    assert rep["rows_contested"] == 2 and rep["drop_gold"] and rep["gold_train_rows_dropped"] >= 1


def _write_gold(tmp_path, root):
    _gold(tmp_path, split_root=root)
    return str(tmp_path / "out" / "gold.csv")


# ---------------------------------------------------------------------------------------------------------------------
# reliability (#12)
# ---------------------------------------------------------------------------------------------------------------------

def _old_rawlabels(tmp_path):
    raw = tmp_path / "rawlabels"
    shutil.copytree(_fix("rawlabels"), raw)
    for f in os.listdir(raw):
        os.utime(raw / f, (OLD, OLD))
    rel = tmp_path / "reliability"
    shutil.copytree(_fix("reliability"), rel)
    return raw, rel


def test_reliability_target_vs_hand_value():
    assert adj.reliability_target(3.2, 0.9) == "correct"  # (3.2+1)/(4.1+2) = 0.689
    assert ftc.smoothed_rate(3.2, 0.9) == pytest.approx(4.2 / 6.1)
    assert adj.reliability_target(0.0, 1.7) == "incorrect" and adj.reliability_target(0.9, 0.9) == ""
    assert adj.reliability_target(None, 1.0) == ""
    assert adj.cached_crowd_target(float("nan")) == ("", 0)


def test_reliability_refuses_snapshot_after_rating(tmp_path):
    raw, rel = _old_rawlabels(tmp_path)
    adj.snapshot_check({"created": "2026-10-22T15:38:50+00:00"}, str(raw), "2026-10-27")
    with pytest.raises(ValueError, match="vote weights were created"):
        adj.snapshot_check({"created": "2026-10-28T00:00:00+00:00"}, str(raw), "2026-10-27")
    with pytest.raises(ValueError, match="no created"):
        adj.snapshot_check({}, str(raw), "2026-10-27")
    adj.snapshot_check({"created": "2026-10-22T15:38:50"}, str(raw), "2026-10-27")  # naive times read as UTC
    os.utime(raw / "city-b.csv", None)  # touched now: after the rating
    with pytest.raises(ValueError, match="city-b.csv"):
        adj.snapshot_check({"created": "2026-10-22T15:38:50+00:00"}, str(raw), "2026-10-27")
    _gold(tmp_path)
    with pytest.raises(SystemExit, match="refusing"):
        adj.main(["reliability", "--gold", str(tmp_path / "out" / "gold.csv"), "--vote-weights",
                  str(rel / "vote_weights.csv"), "--rawlabels", str(raw), "--public-cities", PUBLIC,
                  "--out-dir", str(tmp_path / "agree")])


def test_reliability_coverage_excludes_private_and_unvoted(tmp_path):
    raw, rel = _old_rawlabels(tmp_path)
    g = adj.reliability_frame(_gold(tmp_path), pd.read_csv(rel / "vote_weights.csv"), str(raw), {"city-a", "city-b"})
    g = g.set_index("label_uid")
    assert not g.loc["city-c:2", "covered"]  # private, though in the cache and the weights
    assert not g.loc["city-a:1", "covered"]  # public, no human vote in the cache
    assert g.loc["city-a:5", "covered"] and g.loc["city-a:5", "crowd_cache"] == "correct"  # the AI vote is ignored
    assert g.loc["city-a:4", "crowd_cache"] == "correct" and g.loc["city-a:4", "reliability_weighted"] == "incorrect"
    assert g.loc["city-a:104", "crowd_cache"] == "" and g.loc["city-a:104", "reliability_weighted"] == ""


def test_reliability_table_layout_matches_agreement(tmp_path, capsys):
    raw, rel = _old_rawlabels(tmp_path)
    _gold(tmp_path)
    adj.main(["reliability", "--gold", str(tmp_path / "out" / "gold.csv"), "--vote-weights",
              str(rel / "vote_weights.csv"), "--rawlabels", str(raw), "--public-cities", PUBLIC,
              "--out-dir", str(tmp_path / "agree"), "--n-boot", "50"])
    r = pd.read_csv(tmp_path / "agree" / "reliability.csv")
    agreement_cols = list(adj.agreement_rows(_m([{"expert": "correct"}]), n_boot=0).columns)
    assert agreement_cols == list(r.columns[:len(agreement_cols)])
    assert set(r.model) == set(adj.RELIABILITY_MODELS.values())
    allw = r[(r.model == "reliability-weighted crowd") & (r.label_type == "all")].iloc[0]
    assert allw.n_covered < allw.n_gold_decided
    md = (tmp_path / "agree" / "reliability.md").read_text()
    assert md.splitlines()[0] == adj.agreement_markdown(r).splitlines()[0]
    assert "city-c" not in md and "city-a:" not in md


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def test_report_splices_blocks_and_privacy_check_fails_on_private_name(tmp_path, stub):
    gold, root, crops, out = _scored(tmp_path, stub, no_reason=True)
    agree = tmp_path / "agree"
    adj.main(["agree", "--gold", str(gold), "--scores", str(out), "--models", "stub", "--out-dir", str(agree),
              "--n-boot", "0"])
    doc = tmp_path / "doc.md"
    doc.write_text("# t\n\n" + "\n\n".join(adj.BEGIN.format(name=b) + "\nold\n" + adj.END.format(name=b)
                                            for b in adj.BLOCKS) + "\n\nend\n")
    argv = ["report", "--agree-dir", str(agree), "--gold-summary", str(tmp_path / "out" / "gold_summary.json"),
            "--scores", str(out), "--cityparams", CONF, "--public-cities", PUBLIC, "--out", str(tmp_path / "rep"),
            "--doc", str(doc)]
    adj.main(argv)
    text = doc.read_text()
    assert "\nold\n" not in text and text.endswith("end\n") and "| Model | Type | n |" in text
    assert "_Not yet run._" in text  # no reliability.csv yet
    prov = json.load(open(tmp_path / "rep" / "provenance.json"))
    assert prov["prompt_template_sha256"] == TEMPLATE_SHA and "stub/gold.csv" in prov["scores"]
    for f in os.listdir(tmp_path / "rep"):
        assert "city-c" not in (tmp_path / "rep" / f).read_text()
    # a private deployment named in a generated block stops the report before anything is written
    pd.read_csv(agree / "strata.csv").assign(stratum="city-c").to_csv(agree / "strata.csv", index=False)
    shutil.rmtree(tmp_path / "rep")
    before = doc.read_text()
    with pytest.raises(SystemExit, match=r"private deployment.*city-c"):
        adj.main(argv)
    assert doc.read_text() == before and not (tmp_path / "rep").exists()
    with pytest.raises(ValueError, match="no generated block"):
        adj.splice("no markers", {"agreement": "x"})


def test_repo_doc_has_every_generated_block():
    text = open(adj.DOC).read()
    for b in adj.BLOCKS:
        assert adj.BEGIN.format(name=b) in text and adj.END.format(name=b) in text
    assert "Status: not yet run" in text
    assert "blind" not in text.lower()


def test_build_parser_subcommands():
    p = adj.build_parser()
    for cmd in ("gold", "crops", "stage-list", "prompt", "score", "agree", "disagreements", "relabel", "variants",
                "reliability", "report"):
        assert cmd in p._subparsers._group_actions[0].choices
