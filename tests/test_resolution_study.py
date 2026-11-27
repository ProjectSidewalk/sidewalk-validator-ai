"""analysis/resolution_study.py (docs/resolution-2026-11.md): the registered constants against the doc, the tags,
the run-consistency check, the rule's branches, the power table, and the report on a tiny synthetic run directory."""
import doctest
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from analysis import resolution_study as rs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(REPO, "docs", "resolution-2026-11.md")
PRIVATE = "hiddenvale-xx"   # a made-up deployment the public list leaves out
PUBLIC = ["amsterdam", "chicago-il", "seattle-wa", "pubtown-wa"]
TEXTURE_TAG = {t: s for t, _, s in rs.TEXTURE_SLICES}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_doctests():
    assert doctest.testmod(rs).failed == 0


def test_the_docs_rule_block_equals_the_constant():
    text = open(DOC).read()
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    rules = [json.loads(b) for b in blocks if '"min_delta"' in b]
    assert len(rules) == 1, "the doc must hold exactly one fenced json rule block"
    assert rules[0] == rs.DECISION_RULE


def test_registered_constants():
    assert rs.SEEDS == (42, 43, 44)
    assert rs.DECISION_RULE["min_delta"] == 0.015 and rs.DECISION_RULE["min_target_wins"] == 2
    assert rs.DECISION_RULE["guard_floor"] == -0.005
    assert set(rs.TARGET_TYPES) == {"Obstacle", "SurfaceProblem", "NoCurbRamp"} and rs.GUARD_TYPE == "CurbRamp"
    assert {s for _, _, s in rs.TEXTURE_SLICES} == {"brick/cobblestone", "missing tactile warning", "paint fading"}
    assert rs.ARMS == (448,) and rs.LATER_ARMS == (518,) and rs.BASELINE_RES == 224


@pytest.mark.parametrize("res,seed", [(224, 42), (448, 43), (518, 44)])
def test_tags_round_trip(res, seed):
    assert rs.parse_tag(rs.make_tag(res, seed)) == {"res": res, "seed": seed}
    assert rs.parse_tag(rs.make_tag(res, seed, "smoke-res-"), "smoke-res-") == {"res": res, "seed": seed}
    assert rs.parse_tag(rs.make_tag(res, seed, "smoke-res-")) is None


def test_the_texture_slices_exist_in_the_repos_tag_vocabulary():
    from analysis import slices as sl
    assert sl.parse_tags('["brick/cobblestone", "paint fading"]') == ["brick/cobblestone", "paint fading"]


# ---------------------------------------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------------------------------------

def cmp_(delta=0.03, lo=0.005, hi=0.05, test=0.0, complete=True, texture=()):
    return {"complete": complete, "heldout": {"delta": delta, "lo": lo, "hi": hi}, "mean_d_test_auc": test,
            "texture": list(texture), "arm_res": 448}


def per_type(targets=("win", "win", "win"), guard=0.0, falls=False):
    shapes = {"win": cmp_(), "none": cmp_(0.01, -0.01, 0.03), "reg": cmp_(-0.03, -0.05, -0.01),
              "inc": cmp_(complete=False)}
    out = {t: shapes[s] for t, s in zip(rs.TARGET_TYPES, targets)}
    out[rs.GUARD_TYPE] = cmp_(guard, guard - 0.02, guard + 0.02,
                              texture=[{"type": rs.GUARD_TYPE, "slice": "missing tactile warning", "split": "test",
                                        "evaluable": True, "falls": falls}])
    out[rs.TEXTURE_TYPE] = cmp_(0.0, -0.02, 0.02)
    return out


def test_decide_branches():
    assert rs.decide_arm(per_type())["status"] == "WIN"
    assert rs.decide_arm(per_type(("win", "win", "none")))["status"] == "WIN"
    d = rs.decide_arm(per_type(("win", "none", "none")))
    assert d["status"] == "NO WIN" and "1 of 3" in d["reasons"][0]
    assert rs.decide_arm(per_type(guard=-0.006))["status"] == "NO WIN"
    assert rs.decide_arm(per_type(guard=-0.005))["status"] == "WIN"   # the floor itself passes
    d = rs.decide_arm(per_type(falls=True))
    assert d["status"] == "NO WIN" and "missing tactile warning" in d["reasons"][0]


def test_incomplete_holds_only_while_a_win_is_still_possible():
    assert rs.decide_arm(per_type(("win", "inc", "none")))["status"] == "HELD"
    assert rs.decide_arm(per_type(("none", "inc", "none")))["status"] == "NO WIN"   # 2 wins are out of reach
    p = per_type()
    p[rs.GUARD_TYPE]["complete"] = False
    assert rs.decide_arm(p)["status"] == "HELD"
    p = per_type(("win", "inc", "win"), guard=-0.01)
    assert rs.decide_arm(p)["status"] == "NO WIN"   # the guard already failed


def test_a_win_proposes_448_except_for_a_regressing_type():
    d = rs.decide_arm(per_type(("win", "win", "reg")))
    assert d["status"] == "WIN"
    assert d["proposal"][rs.TARGET_TYPES[2]] == 224 and d["proposal"][rs.TARGET_TYPES[0]] == 448
    assert d["proposal"][rs.GUARD_TYPE] == 448


def test_the_target_test_floor_removes_a_win():
    assert rs.target_verdict(cmp_(test=-0.005))[0] == "WIN"
    assert rs.target_verdict(cmp_(test=-0.0051))[0] == "NO GAIN"
    assert rs.target_verdict(cmp_(0.0149))[0] == "NO GAIN"
    assert rs.target_verdict(cmp_(0.03, 0.0, 0.06))[0] == "NO GAIN"   # the interval must exclude zero


def test_recommend_prefers_448_unless_518_clears_the_margin():
    win = rs.decide_arm(per_type())
    no = rs.decide_arm(per_type(("none", "none", "none")))
    assert rs.recommend({448: no}).startswith("keep 224")
    assert rs.recommend({448: win}).startswith("adopt 448")
    big = per_type()
    for t in rs.TARGET_TYPES:
        big[t] = cmp_(0.034, 0.01, 0.06)   # summed D +0.012 over 448's
    assert rs.recommend({448: win, 518: rs.decide_arm(big)}).startswith("adopt 518")
    close = per_type()
    for t in rs.TARGET_TYPES:
        close[t] = cmp_(0.032, 0.01, 0.06)   # summed D +0.006
    assert rs.recommend({448: win, 518: rs.decide_arm(close)}).startswith("adopt 448")
    assert rs.recommend({448: rs.decide_arm(per_type(("win", "inc", "win")))}).startswith("held")


def test_power_rates_rise_with_seeds_and_effect():
    p = rs.power_table(n_sim=400)
    rows = {(r["type"], r["seeds"]): r for r in p["rows"]}
    for t in rs.TARGET_TYPES:
        assert rows[(t, 3)]["0.000"] < 0.05   # the rule's false-win rate at no effect
        assert rows[(t, 5)]["0.050"] >= rows[(t, 3)]["0.050"] >= rows[(t, 3)]["0.015"]
    assert "| study (>= 2 of 3) |" in rs.power_markdown(p)


def test_at_least():
    assert rs.at_least([1, 1, 0], 2) == 1.0 and abs(rs.at_least([0.9, 0.9, 0.9], 3) - 0.729) < 1e-12


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic $KEEP and split dir
# ---------------------------------------------------------------------------------------------------------------------

CITIES = {"train": ["amsterdam", "chicago-il", "seattle-wa"], "val": ["amsterdam", "chicago-il", "seattle-wa"],
          "test": ["amsterdam", "chicago-il", "seattle-wa"], "test_heldout": ["pubtown-wa", PRIVATE]}
N = {"train": 60, "val": 60, "test": 300, "test_heldout": 600}


def make_split(rng, label_type):
    rows, i = [], 0
    for sp, n in N.items():
        for k in range(n):
            c = CITIES[sp][k % len(CITIES[sp])]
            cls = "incorrect" if rng.random() < 0.45 else "correct"
            tags = ["steep"] if k % 2 else [TEXTURE_TAG.get(label_type, "narrow"), "steep"]
            rows.append({"label_uid": f"{c}:{i}", "city": c, "label_id": i, "pano_id": f"p{i // 2}",
                         "label_type": label_type, "label_class": cls, "crop_path": f"{cls}/{c}_{i}.webp",
                         "split": sp, "hard": 0, "severity": 1 + k % 3, "tags": json.dumps(tags)})
            i += 1
    return pd.DataFrame(rows)


def make_pred(split, rng, signal, shared=None):
    """Scores with ``signal`` separation; ``shared`` (one noise draw per row, reused by every run of a type) plus a
    small per-run jitter keeps the seed spread small, as seeds of one recipe are."""
    ev = split[split.split.isin(["val", "test", "test_heldout"])]
    y = (ev.label_class == "incorrect").to_numpy()
    noise = rng.normal(0, 1, len(ev)) if shared is None else shared + rng.normal(0, 0.15, len(ev))
    p = 1 / (1 + np.exp(-(signal * (2 * y - 1) + noise)))
    return pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                         "p_correct": 1 - p, "p_incorrect": p})


def write_run(keep, lt, tag, split_sha, seed, res, pred, **edits):
    from backbones import preprocess_sha256, preprocess_spec
    d = keep / lt / "runs" / tag
    d.mkdir(parents=True)
    pred.to_csv(d / f"predictions_{tag}.csv", index=False)
    (d / f"metrics_{tag}.json").write_text("{}")
    key = {"label_type": lt, "split_sha256": split_sha, "seed": seed, "target": "hard"}
    if res != 224:
        key["input_res"] = res
    (d / "run_key.json").write_text(json.dumps(key))
    # the fields of training_run.json's backbone block that consistency reads (no hub or weights access)
    backbone = {"key": "dinov2-large", "random_init": False, "preprocess": preprocess_spec("dinov2-large", res),
                "preprocess_sha256": preprocess_sha256("dinov2-large", res)}
    run = {"seed": seed, "epochs": 25, "learning_rate": 1e-5, "best_metric": "macro_f1",
           "model_accepts_loss_kwargs": False, "input_res": res, "gradient_checkpointing": False,
           "started_at": "2026-11-27T00:00:00Z", "finished_at": "2026-11-27T01:00:00Z", "resumed_from": None,
           "peak_gpu_memory_bytes": 18 * 2**30, "backbone": backbone,
           "versions": {"torch": "2.7.1+cu126", "transformers": "4.56.2", "pandas": "2.3.1"}}
    run.update(edits)
    (d / "training_run.json").write_text(json.dumps(run))
    (d / "trainer_state.json").write_text(json.dumps({"max_steps": 600, "log_history": []}))
    (d / "gpu.txt").write_text("NVIDIA L40S, 46068 MiB\n")
    return d


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    split_dir, keep = tmp_path / "split", tmp_path / "keep"
    inputs = {"files": {}, "split_sha256": {}}
    for lt in rs.TYPES:
        (split_dir / lt).mkdir(parents=True)
        split = make_split(rng, lt)
        split.to_csv(split_dir / lt / "split.csv", index=False)
        s = sha(split_dir / lt / "split.csv")
        inputs["split_sha256"][lt] = s
        make_pred(split, rng, 1.0).to_csv(split_dir / lt / "predictions.csv", index=False)
        lift = 1.6 if lt in rs.TARGET_TYPES else 1.0
        shared = rng.normal(0, 1, int(split.split.isin(["val", "test", "test_heldout"]).sum()))
        for seed in rs.SEEDS:
            write_run(keep, lt, rs.make_tag(224, seed), s, seed, 224, make_pred(split, rng, 1.0, shared))
            write_run(keep, lt, rs.make_tag(448, seed), s, seed, 448, make_pred(split, rng, lift, shared))
        # runs that must be flagged, never used
        write_run(keep, "CurbRamp", "res-r518", "0" * 64, 42, 518, make_pred(split, rng, 1.0)) if lt == "CurbRamp" else None
    write_run(keep, "Obstacle", "res-r518-s43", inputs["split_sha256"]["Obstacle"], 43, 224,
              make_pred(make_split(np.random.default_rng(0), "Obstacle"), rng, 1.0))
    inp = tmp_path / "inputs.json"
    inp.write_text(json.dumps(inputs))
    pub = tmp_path / "public.txt"
    pub.write_text("\n".join(PUBLIC) + "\n")
    return {"split_dir": split_dir, "keep": keep, "inputs": inp, "public": pub, "tmp": tmp_path}


def test_consistency_flags_size_hash_and_recipe(synthetic):
    s = json.loads(synthetic["inputs"].read_text())["split_sha256"]
    d = synthetic["keep"] / "CurbRamp" / "runs"
    assert rs.consistency(d / "res-r448-s43", "res-r448-s43", rs.parse_tag("res-r448-s43"), s["CurbRamp"]) == []
    assert rs.consistency(d / "res-r224", "res-r224", rs.parse_tag("res-r224"), s["CurbRamp"]) == []
    assert "split_sha256" in rs.consistency(d / "res-r518", "res-r518", rs.parse_tag("res-r518"), s["CurbRamp"])[0]
    o = synthetic["keep"] / "Obstacle" / "runs" / "res-r518-s43"
    why = rs.consistency(o, "res-r518-s43", rs.parse_tag("res-r518-s43"), s["Obstacle"])
    assert any("input_res" in w for w in why) and any("crops 224, not 518" in w for w in why)
    run_dir = d / "res-r448-s44"
    path = run_dir / "training_run.json"
    orig = path.read_text()
    for edit, reason in ((lambda r: r.update(learning_rate=3e-5), "learning_rate"),
                         (lambda r: r.update(epochs=1), "epochs 1"),
                         (lambda r: r["backbone"].update(key="dinov2-reg-large"), "backbone dinov2-reg-large"),
                         (lambda r: r["backbone"].update(random_init=True), "random_init"),
                         (lambda r: r.update(model_accepts_loss_kwargs=True), "model_accepts_loss_kwargs"),
                         (lambda r: r.update(versions={"transformers": "4.55.0"}), "transformers 4.55.0")):
        rec = json.loads(orig)
        edit(rec)
        path.write_text(json.dumps(rec))
        got = rs.consistency(run_dir, "res-r448-s44", rs.parse_tag("res-r448-s44"), s["CurbRamp"])
        path.write_text(orig)
        assert len(got) == 1 and got[0].startswith(reason), (reason, got)


def run_report(synthetic, out, *extra):
    return rs.main(["report", "--split-dir", str(synthetic["split_dir"]), "--runs-dir", str(synthetic["keep"]),
                    "--inputs", str(synthetic["inputs"]), "--public-cities", str(synthetic["public"]),
                    "--out", str(out), "--n-boot", "40", *extra])


EXPECTED = {"decision.md", "decision.json", "runs.md", "slices.md", "cost.md", "env.md", "gate.md", "provenance.json"}


def test_report_on_a_clear_win(synthetic, capsys):
    out = synthetic["tmp"] / "out"
    assert run_report(synthetic, out) == 0
    printed = capsys.readouterr().out
    assert set(os.listdir(out)) == EXPECTED
    text = "".join(open(out / n).read() for n in EXPECTED)
    assert PRIVATE not in text and PRIVATE not in printed
    assert re.search(r"private city A \((US|outside the US)\)", (out / "runs.md").read_text())
    d = json.loads((out / "decision.json").read_text())
    assert d["rule"] == rs.DECISION_RULE
    assert d["decisions"]["448"]["status"] == "WIN", d["decisions"]["448"]["reasons"]
    assert d["recommendation"].startswith("adopt 448")
    for t in rs.TARGET_TYPES:
        c = d["comparisons"][f"{t}:448"]
        assert c["seeds_arm"] == list(rs.SEEDS) and c["heldout"]["delta"] > 0.015 and c["heldout"]["lo"] > 0
    tex = {(x["type"], x["split"]): x for x in d["decisions"]["448"]["texture"]}
    assert set(tex) == {(t, sp) for t, _, _ in rs.TEXTURE_SLICES for sp in ("test", "test_heldout")}
    assert all(x["evaluable"] and x["train_n"] == 30 for x in tex.values())
    # the flagged runs are listed and not counted
    runs = (out / "runs.md").read_text()
    assert "`res-r518`" in runs and "`res-r518-s43`" in runs
    # slices carry training counts, and every type x severity / type x tag is present
    sl = (out / "slices.md").read_text()
    assert "| SurfaceProblem | 448 | tag | brick/cobblestone (texture) | test | 30 |" in sl
    assert "| Obstacle | 448 | severity | 2 |" in sl
    # the env readout compares the 224 rerun with round 3 at seed 42 only (one reference file)
    assert (out / "env.md").read_text().count("| 42 |") == len(rs.TYPES)
    prov = json.loads((out / "provenance.json").read_text())
    for name, h in prov["outputs"].items():
        assert sha(out / name) == h
    assert "split-dir/CurbRamp/split.csv" in prov["files_read"]
    assert not any(os.path.isabs(k) or str(synthetic["tmp"]) in k for k in prov["files_read"])
    assert str(synthetic["tmp"]) not in json.dumps(prov)


def test_report_holds_with_a_missing_seed(synthetic):
    import shutil
    shutil.rmtree(synthetic["keep"] / "NoCurbRamp" / "runs" / "res-r448-s44")
    shutil.rmtree(synthetic["keep"] / "Obstacle" / "runs" / "res-r448-s44")
    out = synthetic["tmp"] / "out"
    assert run_report(synthetic, out) == 0
    d = json.loads((out / "decision.json").read_text())
    assert d["decisions"]["448"]["status"] == "HELD" and d["recommendation"].startswith("held")


def test_report_refuses_a_split_that_is_not_pinned(synthetic):
    p = synthetic["split_dir"] / "Crosswalk" / "split.csv"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(SystemExit, match="not the pinned"):
        run_report(synthetic, synthetic["tmp"] / "out")


def test_bench_runs_on_a_tiny_config():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    arch = {"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
            "patch_size": 14, "image_size": 518}
    out = rs.bench((224, 448), device="cpu", batches=(1, 2), warmup=0, reps=1, arch=arch, image_px=64)
    assert [(r["res"], r["batch"]) for r in out["rows"]] == [(224, 1), (224, 2), (448, 1), (448, 2)]
    assert all(r["total_ms_per_crop"] > 0 for r in out["rows"]) and out["rows"][2]["tokens"] == 1024
    rows = rs.nightly_rows([out], 10)
    assert len(rows) == 2 and rows[0]["crops_per_night"] == 8000
