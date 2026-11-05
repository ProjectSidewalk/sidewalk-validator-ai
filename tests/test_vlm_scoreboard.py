"""Tests for analysis/vlm_scoreboard.py (issue #63). CPU only, synthetic staging directory, no network."""
import doctest
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from analysis import vlm_scoreboard as sb

PUBLIC = ["seattle-wa", "chicago-il", "tucson-az"]
PRIVATE_TEST, PRIVATE_HELDOUT = ["secretville", "hiddenburg-tx"], ["faraway", "nearby-ok"]
CITIES = PUBLIC[:2] + PRIVATE_TEST + ["tucson-az"] + PRIVATE_HELDOUT
TAGS = ['["narrow"]', '["narrow","pole"]', "[]", None, '["pole"]']


def _splits(n_per=60):
    rows = []
    for t in sb.LABEL_TYPES:
        for i in range(n_per):
            c = CITIES[i % len(CITIES)]
            rows.append({"label_uid": f"{c}:{sb.LABEL_TYPES.index(t)}{i:04d}", "city": c, "label_type": t,
                         "label_class": "incorrect" if (i // len(CITIES)) % 2 else "correct",
                         "split": "test_heldout" if c in ("tucson-az", *PRIVATE_HELDOUT) else "test",
                         "severity": [1.0, 2.0, 3.0, None][i % 4], "tags": TAGS[i % len(TAGS)]})
    return pd.DataFrame(rows)


def _scores(s, signal, seed):
    rng = np.random.default_rng(seed)
    y = (s.label_class == "incorrect").to_numpy()
    p = np.clip(0.5 + signal * (y - 0.5) + rng.normal(0, 0.25, len(y)), 0.001, 0.999)
    return pd.DataFrame({"label_uid": s.label_uid, "model": "m", "variant": "definition", "p_incorrect": p})


def _stage(root):
    s = _splits()
    os.makedirs(root / "splits")
    for t, g in s.groupby("label_type"):
        g.to_csv(root / "splits" / f"{t}.csv", index=False)
    os.makedirs(root / "train")
    for t, g in s.groupby("label_type"):             # stand-in train rows, no ids: a different mix from held-out
        g = pd.concat([g, g[g.severity == 1]])                 # severity 1 twice, so n_train != n
        g[["label_type", "label_class", "severity", "tags"]].to_csv(root / "train" / f"{t}.csv", index=False)
    legs = {"round3": (0.8, s), "production": (0.6, s), "gemma4-31b": (0.4, s),
            "qwen3vl-8b-mlx4": (0.5, s.groupby("label_type").head(20))}
    for i, (leg, (signal, rows)) in enumerate(legs.items()):
        d = root / "scores" / leg
        os.makedirs(d)
        sc = _scores(rows, signal, i).assign(label_type=rows.label_type.values)
        sc = sc.sample(frac=1, random_state=100 + i)     # leg rows in their own order, never the splits' order
        for t, g in sc.groupby("label_type"):
            g.drop(columns="label_type").to_csv(d / f"{t}.csv", index=False)
    pd.DataFrame([{"model": "gemma4-31b", "gpu": "H200", "n_gpus": 1, "batch_size": 1, "n_crops": 50,
                   "median_s_per_crop": 0.4, "p90_s_per_crop": 0.5, "source": "test"},
                  {"model": "gemma4-31b", "gpu": "H200", "n_gpus": 1, "batch_size": 16, "n_crops": 1000,
                   "median_s_per_crop": 0.05, "p90_s_per_crop": 0.06, "source": "test"}]
                 ).to_csv(root / "scores" / "timing.csv", index=False)
    json.dump({"molmo2-8b": {"*": "loader failed on transformers 4.57 (see runs.md)"}},
              open(root / "scores" / "notes.json", "w"))
    return s


def _doc(path, prose=""):
    blocks = "\n\n".join(f"{sb.BEGIN.format(name=n)}\n{sb.END.format(name=n)}" for n in
                         ("short_answer", "board", "per_type", "paired", "per_deployment", "timing", "coverage",
                          "by_severity", "by_tag"))
    path.write_text(f"# Test board\n\n{prose}\n\n{blocks}\n")


def _readme(path):
    path.write_text(f"# Test readme\n\nScores:\n\n{sb.BEGIN.format(name=sb.README_BLOCK)}\n{sb.END.format(name=sb.README_BLOCK)}\n")


def _prose(aggs):
    f = sb.facts(aggs)
    return (f"The board has {f['legs']} legs, {f['full_legs']} of them scored all {f['n_total']:,} held-out crops. "
            f"Round 3 leads the best VLM ({f['best_vlm']}) by {f['gap']:.3f} macro-AUC on the crops both scored.")


@pytest.fixture
def board(tmp_path):
    stage, out, doc, readme = tmp_path / "stage", tmp_path / "out", tmp_path / "doc.md", tmp_path / "README.md"
    _stage(stage)
    pub = tmp_path / "public.txt"
    pub.write_text("\n".join(PUBLIC))
    base = ["build", "--scores-dir", str(stage), "--out-dir", str(out), "--doc", str(doc), "--readme", str(readme),
            "--public-cities", str(pub), "--n-boot", "50"]
    _doc(doc)
    _readme(readme)
    sb.main(base)                                   # aggregates, figures (prose not yet written)
    _doc(doc, _prose(sb.read_aggregates(str(out))))
    sb.main(base)                                   # fills the blocks
    return {"stage": stage, "out": out, "doc": doc, "readme": readme, "base": base, "pub": pub}


def test_doctests():
    res = doctest.testmod(sb)
    assert res.failed == 0 and res.attempted > 0


def test_rank_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)                # many ties
    assert sb.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_boot_auc_pairs_columns_and_brackets_point():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    a = y + rng.normal(0, 1, 400)
    b = sb.boot_auc(y, np.c_[a, a], 200, np.random.default_rng(0))
    assert np.allclose(b[:, 0], b[:, 1])            # same draws for both columns
    lo, hi = sb.ci(b[:, 0])
    assert lo < sb.auc(y, a) < hi


def test_macro_auc_is_mean_of_per_type(board):
    a = sb.read_aggregates(str(board["out"]))
    for r in a["board"].itertuples():
        pt = a["per_type"][a["per_type"].leg == r.leg]
        assert r.macro_auc == pytest.approx(pt.auc.mean(), abs=2e-4)
        assert r.macro_auc == pytest.approx(np.mean([getattr(r, f"auc_{t}") for t in sb.LABEL_TYPES]), abs=2e-4)


def test_partial_legs_sort_below_full_and_are_not_ranked(board):
    b = sb.read_aggregates(str(board["out"]))["board"]
    full = b.full.astype(bool).tolist()
    assert full == sorted(full, reverse=True)       # all full rows first
    assert b[b.full].macro_auc.is_monotonic_decreasing
    assert b[~b.full].leg.tolist() == ["qwen3vl-8b-mlx4"]
    assert b[~b.full]["rank"].isna().all() and b[b.full]["rank"].tolist() == [1, 2, 3]
    assert b.leg.iloc[0] == "round3"
    text = board["doc"].read_text()
    assert text.index("partial coverage, not ranked") < text.index("Qwen3-VL-8B (4-bit, Mac)")


def test_coverage_table_says_why(board):
    c = sb.read_aggregates(str(board["out"]))["coverage"]
    incomplete = c[~c.complete.astype(bool)]
    assert incomplete.why.notna().all() and (incomplete.why.str.len() > 0).all()
    assert c[c.leg == "molmo2-8b"].why.str.contains("loader failed").all()        # notes.json
    assert c[c.leg == "qwen3vl-8b-mlx4"].why.str.contains("4-bit, Mac, sample").all()
    assert c[c.leg == "qwen3.8-27b"].why.eq("not run yet").all()                   # roster, not staged
    assert (c.leg == "fewshot8").any()
    assert c[c.leg == "round3"].why.isna().all()


def test_private_deployments_pooled_and_never_named(board):
    out, doc = board["out"], board["doc"]
    assert sb.privacy_problems(str(out), str(doc), set(PUBLIC)) == []
    dep = sb.read_aggregates(str(out))["per_deployment"]
    names = set(dep.deployment)
    assert "private city A (outside the US)" in names and "private city B (US)" in names
    assert "other (2 private deployments)" in names
    blob = doc.read_text() + "".join(open(os.path.join(out, f)).read() for f in os.listdir(out)
                                     if f.endswith((".csv", ".json")))
    for c in PRIVATE_TEST + PRIVATE_HELDOUT:
        assert c not in blob


def test_privacy_scan_catches_a_label_key_and_a_private_city(board, tmp_path):
    out = board["out"]
    pd.DataFrame({"deployment": ["seattle-wa", "secretville"]}).to_csv(out / "leak.csv", index=False)
    (out / "leak2.json").write_text('{"x": "faraway:123"}')
    probs = sb.privacy_problems(str(out), str(board["doc"]), set(PUBLIC))
    assert any("secretville" in p for p in probs) and any("faraway:123" in p for p in probs)


def test_check_passes_when_current_and_fails_on_drift(board, capsys):
    base = board["base"]
    sb.main(base + ["--check"])                     # full mode: staging dir present
    # CI mode: no staging dir, doc against the committed aggregates
    ci = [a if a != str(board["stage"]) else str(board["stage"]) + "-absent" for a in base]
    sb.main(ci + ["--check"])
    # a CSV drifts -> full mode fails, CI mode does not see per-crop data but still checks the doc
    board_csv = board["out"] / "board.csv"
    good = board_csv.read_text()
    df = pd.read_csv(board_csv)
    df.loc[0, "macro_auc"] = 0.5
    df.to_csv(board_csv, index=False, float_format=sb.FLOAT)
    with pytest.raises(SystemExit):
        sb.main(base + ["--check"])
    with pytest.raises(SystemExit):                 # the doc no longer matches the committed CSV either
        sb.main(ci + ["--check"])
    board_csv.write_text(good)
    sb.main(base + ["--check"])
    # a generated block edited by hand
    doc = board["doc"]
    text = doc.read_text()
    doc.write_text(text.replace("| 1 |", "| 9 |", 1))
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])
    doc.write_text(text)
    # regenerating restores a hand-edited block
    doc.write_text(text.replace("| 1 |", "| 9 |", 1))
    sb.main(base)
    sb.main(base + ["--check"])
    # the README block: hand-edited -> both modes fail; rebuilt -> current; missing marker -> fails; no file -> skipped
    readme = board["readme"]
    rtext = readme.read_text()
    assert "Round 3" in rtext and "Previous production" in rtext
    readme.write_text(rtext.replace("0.", "9.", 1))
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])
    sb.main(base)
    assert readme.read_text() == rtext
    sb.main(ci + ["--check"])
    readme.write_text("# no block here\n")
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])
    prov = (board["out"] / "provenance.json").read_text()
    with pytest.raises(SystemExit, match="no generated block"):   # build refuses before writing anything
        sb.main(base)
    assert (board["out"] / "provenance.json").read_text() == prov
    readme.unlink()
    with pytest.raises(SystemExit):                 # a missing README is a problem, not a skip
        sb.main(ci + ["--check"])
    none = [a if a != str(readme) else "" for a in ci]
    sb.main(none + ["--check"])                     # --readme '' opts out
    _readme(readme)
    sb.main(base)
    # the privacy scan covers the README, through main
    readme.write_text(readme.read_text() + "\nleak faraway:123\n")
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])
    with pytest.raises(SystemExit):
        sb.main(base)
    _readme(readme)
    sb.main(base)


def test_readme_block_is_round3_production_and_best_full_vlms(board):
    aggs = sb.read_aggregates(str(board["out"]))
    b = aggs["board"]
    # The fixture has one full VLM leg; add four more with distinct AUCs so the sort and the head cut are both tested.
    template = b[b.full.astype(bool) & (b.model_class != sb.PURPOSE)].iloc[0]
    extra = []
    for i, auc in enumerate((0.61, 0.79, 0.70, 0.66)):
        r = template.copy()
        r["leg"], r["display"], r["macro_auc"], r["rank"] = f"extra{i}", f"Extra model {i}", auc, np.nan
        extra.append(r)
    b = pd.concat([b, pd.DataFrame(extra)], ignore_index=True)
    aggs["board"] = b.sample(frac=1, random_state=3)    # the block must sort itself, not rely on board order
    block = sb.render_readme_block(aggs)
    lines = [l for l in block.splitlines() if l.startswith("| ") and not l.startswith("| model")]
    full = b[b.full.astype(bool)]
    purpose = full[full.model_class == sb.PURPOSE].display.tolist()
    vlms = full[full.model_class != sb.PURPOSE].sort_values("macro_auc", ascending=False).display.tolist()
    want = purpose + vlms[:sb.README_VLMS]
    assert [l.split("|")[1].strip().strip("*") for l in lines] == want
    assert all(b[~b.full.astype(bool)].display.map(lambda d: d not in block))     # partial legs never shown
    assert lines[0].split("|")[-2].strip() == "—"                                # round 3 has no Δ
    # the privacy scan covers the README too
    board["readme"].write_text(board["readme"].read_text() + "\nleak faraway:123\n")
    assert any("faraway:123" in p for p in sb.privacy_problems(str(board["out"]), str(board["doc"]), set(PUBLIC),
                                                                 str(board["readme"])))


def test_prose_counts_are_checked(board):
    doc = board["doc"]
    text = doc.read_text()
    a = sb.read_aggregates(str(board["out"]))
    assert sb.prose_problems(text, a) == []
    f = sb.facts(a)
    assert f["legs"] == 4 and f["full_legs"] == 3 and f["best_vlm"] == "Gemma 4 31B"
    bad = text.replace(f"has {f['legs']} legs", "has 5 legs")
    assert any("intro: legs" in p for p in sb.prose_problems(bad, a))
    bad = text.replace(f"{f['full_legs']} of them", "2 of them")
    assert any("intro: full legs" in p for p in sb.prose_problems(bad, a))
    bad = text.replace(f"by {f['gap']:.3f}", "by 0.001")
    assert any("headline gap" in p for p in sb.prose_problems(bad, a))
    bad = text.replace("Round 3 leads the best VLM", "Round 3 beats the best VLM")
    assert any("no sentence states it" in p for p in sb.prose_problems(bad, a))


def test_headline_gap_is_round3_minus_best_full_vlm(board):
    a = sb.read_aggregates(str(board["out"]))
    b = a["board"].set_index("leg")
    assert sb.facts(a)["gap"] == pytest.approx(b.loc["round3", "macro_auc"] - b.loc["gemma4-31b", "macro_auc"],
                                               abs=2e-4)


def test_figures_render_on_agg(board):
    for f in ("leaderboard", "by_type", "roc", "per_deployment", "latency", "by_severity", "by_tag"):
        p = board["out"] / f"{f}.png"
        assert p.exists() and p.stat().st_size > 5000


def test_timing_uses_largest_batch_and_gpu_hours(board):
    a = sb.read_aggregates(str(board["out"]))
    r = a["board"].set_index("leg").loc["gemma4-31b"]
    assert r.s_per_crop == pytest.approx(0.05) and int(r.batch_size) == 16 and r.gpu == "H200"
    t = a["timing"]
    assert t[t.batch_size == 16].gpu_hours_per_1000.iloc[0] == pytest.approx(0.05 * 1000 / 3600, abs=1e-4)


def test_score_outside_the_splits_raises(board):
    d = board["stage"] / "scores" / "gemma4-31b" / "CurbRamp.csv"
    df = pd.read_csv(d)
    pd.concat([df, df.iloc[:1].assign(label_uid="seattle-wa:999999")]).to_csv(d, index=False)
    with pytest.raises(ValueError, match="not in the held-out splits"):
        sb.main(board["base"] + ["--no-figures"])


def test_conflicting_duplicate_scores_raise(board):
    d = board["stage"] / "scores" / "gemma4-31b" / "CurbRamp.csv"
    df = pd.read_csv(d)
    pd.concat([df, df.iloc[:1].assign(p_incorrect=0.123)]).to_csv(d, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        sb.main(board["base"] + ["--no-figures"])


def test_stage_refuses_a_dir_inside_the_repo(tmp_path):
    with pytest.raises(SystemExit):
        sb.main(["stage", "--run-dir", str(tmp_path), "--scores-dir", os.path.join(sb.REPO_DIR, "docs", "x")])


def test_committed_board_is_current_and_private_safe(monkeypatch):
    """The CI drift check: the committed doc against the committed aggregate CSVs (no staging dir needed). Offline,
    so the public list is the one frozen in provenance.json."""
    if not os.path.exists(os.path.join(sb.OUT_DIR, "board.csv")):
        pytest.skip("no committed board yet")
    monkeypatch.setattr(sb, "_live_public", _offline)
    sb.main(["build", "--check", "--scores-dir", os.path.join(sb.REPO_DIR, "no-such-staging-dir")])


def _offline():
    raise OSError("no network in tests")


def test_check_fails_when_a_named_deployment_goes_private(board, monkeypatch, capsys):
    """--check without --public-cities asks the live API; a deployment shown by name that has since gone private
    fails the check in both modes. Offline it falls back to provenance.json and passes."""
    base = [a for a in board["base"] if a not in ("--public-cities", str(board["pub"]))]
    ci = [a if a != str(board["stage"]) else str(board["stage"]) + "-absent" for a in base]
    dep = sb.read_aggregates(str(board["out"]))["per_deployment"]
    assert "seattle-wa" in set(dep.deployment)
    monkeypatch.setattr(sb, "_live_public", lambda: (set(PUBLIC), "live"))
    sb.main(ci + ["--check"])
    sb.main(base + ["--check"])
    monkeypatch.setattr(sb, "_live_public", lambda: (set(PUBLIC) - {"seattle-wa"}, "live"))
    for argv in (ci, base):
        with pytest.raises(SystemExit):
            sb.main(argv + ["--check"])
        assert "'seattle-wa' is no longer public" in capsys.readouterr().out
    monkeypatch.setattr(sb, "_live_public", _offline)
    sb.main(ci + ["--check"])
    assert "frozen in provenance.json" in capsys.readouterr().out


def test_unscored_vllm_rows_are_dropped_and_explained(board):
    """answer_floor == "both" rows (neither answer in the top 20, a 0.5 placeholder) are not scores: the leg drops
    them, shows as partial, and the coverage table says how many and why."""
    d = board["stage"] / "scores" / "gemma4-31b"
    for t in ("CurbRamp", "Obstacle"):
        df = pd.read_csv(d / f"{t}.csv").assign(answer_floor="")
        if t == "CurbRamp":
            df.loc[df.index[:3], ["answer_floor", "p_incorrect"]] = ["both", 0.5]
            df.loc[df.index[3], "answer_floor"] = "no"             # one side floored: still a score
        df.to_csv(d / f"{t}.csv", index=False)
    sb.main(board["base"] + ["--no-figures"])
    a = sb.read_aggregates(str(board["out"]))
    r = a["board"].set_index("leg").loc["gemma4-31b"]
    assert not bool(r.full) and int(r.n) == int(r.n_total) - 3
    c = a["coverage"].set_index(["leg", "label_type"])
    assert c.loc[("gemma4-31b", "CurbRamp"), "why"] == "3 rows unscored: neither yes nor no in the top-20 log-probs"
    assert not bool(c.loc[("gemma4-31b", "CurbRamp"), "complete"])
    assert bool(c.loc[("gemma4-31b", "Obstacle"), "complete"])
    legs, unscored = sb.load_scores(str(board["stage"]), sb.load_splits(str(board["stage"])))
    assert unscored == {"gemma4-31b": {"CurbRamp": 3}} and len(legs["round3"]) == int(r.n_total)


def test_run_speed_prefers_the_full_run_with_most_crops_on_a_tie():
    t = pd.DataFrame([
        {"model": "m", "gpu": "g-full-big", "batch_size": 1, "n_crops": 13144, "median_s_per_crop": 0.26,
         "source": "full run"},
        {"model": "m", "gpu": "g-timing", "batch_size": 1, "n_crops": 50, "median_s_per_crop": 0.96,
         "source": "batch-1 timing run"},
        {"model": "m", "gpu": "g-full-small", "batch_size": 1, "n_crops": 1988, "median_s_per_crop": 0.30,
         "source": "full run"},
        {"model": "other", "gpu": "x", "batch_size": 64, "n_crops": 9, "median_s_per_crop": 9.0, "source": "full run"},
    ])
    for order in (t, t.iloc[::-1], t.iloc[[1, 0, 3, 2]]):
        assert sb.run_speed(order.reset_index(drop=True), "m") == (0.26, "g-full-big", 1)
    bigger = pd.concat([t, pd.DataFrame([{"model": "m", "gpu": "g-b16", "batch_size": 16, "n_crops": 50,
                                          "median_s_per_crop": 0.1, "source": "timing"}])])
    assert sb.run_speed(bigger, "m") == (0.1, "g-b16", 16)            # the largest batch still comes first


def test_slices_cover_full_legs_only_and_count_each_tag(board):
    a = sb.read_aggregates(str(board["out"]))
    sev, tag = a["by_severity"], a["by_tag"]
    full = set(a["board"][a["board"].full.astype(bool)].leg)
    assert set(sev.leg) == full and set(tag.leg) == full           # the partial Mac leg is not sliced
    r = sev[(sev.leg == "round3") & (sev.label_type == "CurbRamp")]
    assert r.severity.tolist() == ["1", "2", "3", "(none)"] and r.n.sum() == 60
    assert r.n_train.tolist() == [30, 15, 15, 15]                 # counted from the train rows, not the held-out
    assert np.allclose(r.incorrect_share, r.n_incorrect / r.n, atol=1e-4) and (r.n_test + r.n_test_heldout == r.n).all()
    # every real tag has 24 of 60 crops, under the 30-crop row rule: only (none) gets a row
    assert set(tag.tag) == {"(none)"}
    assert tag[(tag.leg == "round3") & (tag.label_type == "CurbRamp")].n.iloc[0] == 24     # "[]" and missing
    text = board["doc"].read_text()
    assert "| CurbRamp | (none) |" in text


def test_slice_names_and_rows_on_a_small_frame(monkeypatch):
    monkeypatch.setattr(sb, "SLICE_MIN_N", 4)
    monkeypatch.setattr(sb, "SLICE_MIN_CLASS", 2)
    s = pd.DataFrame({"label_uid": [f"x:{i}" for i in range(8)], "label_type": "CurbRamp",
                      "y": [0, 1, 0, 1, 0, 1, 0, 1], "severity": [1, 1, 1, 1, 2, None, 2, 2],
                      "tags": ['["a","b"]', '["a"]', '["a","b"]', '["a","b"]', '["b"]', "[]", '["c"]', '["b"]']})
    assert sb.slice_names(s, "severity") == {"CurbRamp": ["1", "2", "(none)"]}
    assert sb.slice_names(s, "tag") == {"CurbRamp": ["b", "a", "(none)"]}    # most common first; "c" is too thin
    # tag "a" (rows 0-3) is all test, three of its four crops in city p
    s = s.assign(city=["p", "p", "p", "q", "p", "q", "q", "q"], split=["test"] * 4 + ["test_heldout"] * 4)
    legs = {"good": s.assign(p_incorrect=s.y * 0.8 + 0.1), "bad": s.assign(p_incorrect=1 - (s.y * 0.8 + 0.1))}
    rows = pd.DataFrame(sb.slice_rows(legs, s, None, "tag", n_boot=20, seed=1, within_min_n=2))
    a = rows[rows.tag == "a"].set_index("leg")
    assert a.loc["good", "auc"] == 1.0 and a.loc["bad", "auc"] == 0.0 and a.loc["good", "n"] == 4
    assert a.loc["good", "top_deployment_share"] == 0.75
    assert a.loc["good", "auc_test"] == 1.0 and np.isnan(a.loc["good", "auc_test_heldout"])
    assert a.loc["good", "n_test"] == 4 and a.loc["good", "n_test_heldout"] == 0
    assert a.loc["good", "within_city_auc"] == 1.0 and a.loc["good", "n_cities"] == 1   # q has one crop of "a"
    rows = pd.DataFrame(sb.slice_rows(legs, s, None, "tag", n_boot=20, seed=1))     # default: cities need 10 rows
    assert np.isnan(rows.within_city_auc).all()
    assert rows.n_train.isna().all()                               # no train rows staged
    assert np.isnan(rows[rows.tag == "(none)"].auc).all()          # one crop: blank


def test_slice_interval_is_the_label_bootstrap_and_ignores_row_order():
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    s = pd.DataFrame({"label_uid": [f"x:{i:03d}" for i in range(80)], "label_type": "CurbRamp", "city": "p",
                      "split": np.where(np.arange(80) % 2, "test", "test_heldout"), "y": y, "severity": 2,
                      "tags": "[]"})
    m = s.assign(p_incorrect=rng.random(80) + 0.4 * y)
    rows = sb.slice_rows({"m": m, "shuffled": m.sample(frac=1, random_state=1)}, s, None, "severity", 200, 7)
    r, sh = rows
    lo, hi = sb.ci(sb.boot_auc(y, m.p_incorrect, 200, sb._slice_rng(7, "severity", "CurbRamp", "2"))[:, 0])
    assert (r["auc_lo"], r["auc_hi"]) == (lo, hi) and lo < r["auc"] < hi
    assert (sh["auc_lo"], sh["auc_hi"]) == (lo, hi)                # same draws whatever order the leg's rows are in
    for sp in sb.SPLITS:                                           # per split: that split's crops only
        k = (s.split == sp).to_numpy()
        assert r[f"auc_{sp}"] == pytest.approx(sb.auc(y[k], m.p_incorrect[k])) and r[f"n_{sp}"] == 40
    assert r["auc_test"] != pytest.approx(r["auc_test_heldout"])


def test_slice_auc_needs_enough_of_each_class(monkeypatch):
    monkeypatch.setattr(sb, "SLICE_MIN_N", 4)
    monkeypatch.setattr(sb, "SLICE_MIN_CLASS", 2)
    s = pd.DataFrame({"label_uid": [f"x:{i}" for i in range(6)], "label_type": "CurbRamp", "city": "p",
                      "split": "test", "y": [1, 1, 1, 1, 1, 0], "severity": 1, "tags": "[]"})
    r = sb.slice_rows({"m": s.assign(p_incorrect=np.linspace(0.9, 0.1, 6))}, s, None, "severity", n_boot=10,
                      within_min_n=2)[0]
    assert r["n"] == 6 and np.isnan(r["auc"]) and np.isnan(r["auc_test"])    # one correct crop: no AUC
    assert np.isnan(r["within_city_auc"])                         # nor within city, though the city has 6 rows
    tr = pd.DataFrame({"label_type": ["CurbRamp"], "label_class": ["correct"], "severity": [1], "tags": ["[]"]})
    r = sb.slice_rows({"m": s.assign(p_incorrect=0.5)}, s, tr, "severity", n_boot=10)[0]
    assert r["n_train"] == 1
    other = tr.assign(label_type="Obstacle")                      # train rows of another type only: blank, not 0
    assert np.isnan(sb.slice_rows({"m": s.assign(p_incorrect=0.5)}, s, other, "severity", n_boot=10)[0]["n_train"])


def test_check_catches_an_edited_slice_cell_in_both_modes(board, monkeypatch):
    base = board["base"]
    monkeypatch.setattr(sb, "SLICE_MIN_N", 5)      # synthetic severity slices: 15 crops, ~10 per split
    monkeypatch.setattr(sb, "SLICE_MIN_CLASS", 2)
    sb.main(base)
    ci = [a if a != str(board["stage"]) else str(board["stage"]) + "-absent" for a in base]
    p = board["out"] / "by_severity.csv"
    good = p.read_text()
    df = pd.read_csv(p, dtype={"severity": str}, keep_default_na=False, na_values=[""])
    i = df.index[(df.leg == "gemma4-31b") & df.auc.notna()][0]
    df.loc[i, "auc"] = min(df.loc[i, "auc_hi"] + 0.05, 0.9999)
    df.to_csv(p, index=False, float_format=sb.FLOAT)
    with pytest.raises(SystemExit):
        sb.main(base + ["--check"])                 # staged: byte comparison
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])                   # CI: AUC outside its own interval
    p.write_text(good)
    df = pd.read_csv(p, dtype={"severity": str}, keep_default_na=False, na_values=[""])
    j = df.index[df.auc_test.notna()][0]
    df.loc[j, "auc_test"] = df.loc[j, "auc_test"] - 0.01   # a column the doc does not show: only staged mode sees it
    df.to_csv(p, index=False, float_format=sb.FLOAT)
    sb.main(ci + ["--check"])
    with pytest.raises(SystemExit):
        sb.main(base + ["--check"])
    p.write_text(good)


@pytest.mark.parametrize("column, delta", [("incorrect_share", 0.001), ("n_test", 1)])
def test_ci_check_runs_the_consistency_checks(board, column, delta):
    """Edits the doc tables do not show (a share off by 0.001 still renders as the same %, n_test is not shown):
    only consistency_problems can catch them in CI mode."""
    ci = [a if a != str(board["stage"]) else str(board["stage"]) + "-absent" for a in board["base"]]
    p = board["out"] / "by_severity.csv"
    good = p.read_text()
    df = pd.read_csv(p, dtype={"severity": str}, keep_default_na=False, na_values=[""])
    df.loc[0, column] = df.loc[0, column] + delta
    df.to_csv(p, index=False, float_format=sb.FLOAT)
    with pytest.raises(SystemExit):
        sb.main(ci + ["--check"])
    p.write_text(good)
    sb.main(ci + ["--check"])


def test_consistency_problems_each_branch():
    ok = pd.DataFrame({"leg": ["a"], "auc": [0.7], "auc_lo": [0.6], "auc_hi": [0.8], "n": [10], "n_incorrect": [4],
                       "incorrect_share": [0.4], "n_test": [6], "n_test_heldout": [4]})
    assert sb.consistency_problems({"by_tag": ok}) == []
    assert "interval" in sb.consistency_problems({"by_tag": ok.assign(auc=0.9)})[0]
    assert "incorrect_share" in sb.consistency_problems({"by_tag": ok.assign(incorrect_share=0.5)})[0]
    assert "sum to n" in sb.consistency_problems({"by_tag": ok.assign(n_test=7)})[0]


def test_stage_writes_train_rows_without_ids(tmp_path):
    run = tmp_path / "run"
    for k, t in enumerate(sb.LABEL_TYPES):
        os.makedirs(run / t)
        rows = [{"label_uid": f"c:{k}{i:03d}", "city": "c", "label_type": t, "label_class": ["correct", "incorrect"][i % 2],
                 "split": ["train", "train", "train", "test", "test_heldout"][i % 5], "severity": 1 + i % 3,
                 "tags": '["x"]', "status": "ok" if i != 0 else "missing", "crop_path": f"{i}.webp"} for i in range(20)]
        d = pd.DataFrame(rows)
        d.to_csv(run / t / "split.csv", index=False)
        held = d[d.split.isin(sb.SPLITS)]
        held[["label_uid"]].assign(p_incorrect=0.5).to_csv(run / t / "predictions.csv", index=False)
    stage = tmp_path / "stage"
    sb.main(["stage", "--run-dir", str(run), "--scores-dir", str(stage)])
    tr = pd.read_csv(stage / "train" / "CurbRamp.csv")
    assert list(tr.columns) == ["label_type", "label_class", "severity", "tags"]   # no label_uid, no city
    assert len(tr) == 11                                          # 12 train rows, one not status ok
    assert {"severity", "tags"} <= set(pd.read_csv(stage / "splits" / "CurbRamp.csv").columns)


def test_paired_rows_use_the_same_draws_for_both_legs():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = rng.random(200) + 0.3 * y
    d = pd.DataFrame({"label_uid": [f"x:{i}" for i in range(200)], "label_type": "CurbRamp", "y": y, "p_incorrect": s})
    r = sb.paired_rows("leg", d, d[["label_uid", "p_incorrect"]], 200, 1)
    assert r[0]["diff"] == 0 and r[0]["diff_lo"] == 0 and r[0]["diff_hi"] == 0   # identical scores: zero width
    noisy = d.assign(p_incorrect=d.p_incorrect + rng.normal(0, 0.05, 200))
    r = sb.paired_rows("leg", noisy, d[["label_uid", "p_incorrect"]], 500, 1)[0]
    lo, hi = sb.ci(sb.boot_auc(y, s, 500, np.random.default_rng(9))[:, 0])
    assert r["diff_hi"] - r["diff_lo"] < (hi - lo) / 2           # pairing: far narrower than one leg's own interval


def test_within_city_and_macro_interval_on_the_board(board):
    a = sb.read_aggregates(str(board["out"]))
    pt = a["per_type"]
    stage = board["stage"]
    sp = sb.load_splits(str(stage))
    legs, _ = sb.load_scores(str(stage), sp)
    g = legs["round3"][legs["round3"].label_type == "CurbRamp"]
    rows, _ = sb.per_type_rows("round3", legs["round3"], 20, 1, within_min_n=3)   # synthetic cities hold ~8 rows
    r = [x for x in rows if x["label_type"] == "CurbRamp"][0]
    exp, nc = sb.vb.within_city_auc(g.y, g.p_incorrect, g.city, 3)
    assert nc > 1 and r["within_city_auc"] == pytest.approx(exp) and r["n_cities"] == nc
    b = a["board"].set_index("leg").loc["round3"]
    assert b.macro_auc_lo < b.macro_auc < b.macro_auc_hi
    width = (pt[pt.leg == "round3"].auc_hi - pt[pt.leg == "round3"].auc_lo).mean()
    assert b.macro_auc_hi - b.macro_auc_lo < width                # averaging five types narrows the interval


def test_old_staging_without_severity_builds_without_slices(tmp_path):
    root = tmp_path / "stage"
    s = _stage(root)
    for t, g in s.groupby("label_type"):
        g.drop(columns=["severity", "tags"]).to_csv(root / "splits" / f"{t}.csv", index=False)
    aggs, *_ = sb.build_aggregates(str(root), set(PUBLIC), n_boot=10)
    assert aggs["by_severity"].empty and aggs["by_tag"].empty
    assert sb.slice_table(aggs["by_tag"], aggs["board"], "tag") == "(no slices staged)"


def test_bold_marks_the_best_cell_and_the_board_is_two_tables(board):
    assert sb.bold_best(["0.915 [0.910, 0.919]", "0.915", "0.899", "—"]) == \
        ["**0.915 [0.910, 0.919]**", "**0.915**", "0.899", "—"]          # ties at the shown precision are all bold
    assert sb.bold_best(["0.05 (A40, 1)", "0.5 (H200, 16)"], higher=False) == ["**0.05 (A40, 1)**", "0.5 (H200, 16)"]
    assert sb.bold_best(["0.9", "—"]) == ["0.9", "—"]                      # nothing to beat
    aggs = sb.read_aggregates(str(board["out"]))
    b = aggs["board"]
    full = b[b.full.astype(bool)]
    top = full.sort_values("macro_auc", ascending=False).iloc[0]
    # README block: no class column, the model name plain, the best macro-AUC and per-type AUCs bold
    block = sb.render_readme_block(aggs)
    assert "| class |" not in block
    row = next(l for l in block.splitlines() if l.startswith(f"| {top.display} |"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[1].startswith("**") and cells[1].endswith("**")
    assert sum(c.startswith("**") for c in cells[2:7]) >= 1
    assert not cells[-1].startswith("**") and "| crops |" not in block         # Δ is never bold; no crops column
    # the doc's board: two tables, bold among the full rows only, partial rows after the rule in both
    doc = sb.render_blocks(aggs)["board"]
    assert doc.count("\n|---|") == 2 and doc.count("partial coverage, not ranked") == 2
    for l in doc.splitlines():
        if "(4-bit, Mac)" in l:
            assert "**" not in l
    first = doc.split("\n\n")[0]
    top_row = next(l for l in first.splitlines() if f"| {top.display} |" in l)
    assert "**" in top_row


def test_bold_in_the_other_tables_and_the_rule(board):
    aggs = sb.read_aggregates(str(board["out"]))
    b = aggs["board"]
    full_names = set(b[b.full.astype(bool)].display)
    blocks = sb.render_blocks(aggs)

    def body(block):
        return [[c.strip() for c in l.split("|")[1:-1]] for l in block.splitlines()[2:] if l.startswith("| ")]

    # paired: the larger of the two AUCs is bold, the difference never
    for cells in body(blocks["paired"]):
        assert sum(c.startswith("**") for c in cells[3:5]) >= 1 and not cells[5].startswith("**")
    # per deployment: one winner per row among the leg columns
    for cells in body(blocks["per_deployment"]):
        assert sum(c.startswith("**") for c in cells[3:]) >= 1
    # per type: every numeric column has a winner within each type, and only full legs carry bold
    rows = body(blocks["per_type"])
    for t in sb.LABEL_TYPES:
        group = [r for r in rows if r[1] == t and r[0] in full_names]
        for j in range(4, 10):
            if sum(sb._value(r[j]) is not None for r in group) >= 2:      # a column of "—" has no winner
                assert any(r[j].startswith("**") for r in group), (t, j)
    assert not any(c.startswith("**") for r in rows if r[0] not in full_names for c in r)
    # slices: the AUC pair and the within-city pair are bolded separately; a slice without within-city values has
    # no bold there but still a bold AUC (the fixture's own slices are all too thin to score, so build two rows)
    def srow(leg, sev, auc, within):
        return dict(leg=leg, label_type="CurbRamp", severity=sev, n=40, incorrect_share=0.5,
                    top_deployment_share=0.6, n_train=np.nan, auc=auc, auc_lo=auc - 0.05, auc_hi=auc + 0.05,
                    within_city_auc=within)
    df = pd.DataFrame([srow(sb.REFERENCE, 1, 0.9, 0.8), srow("production", 1, 0.7, 0.85),
                       srow(sb.REFERENCE, 2, 0.6, np.nan), srow("production", 2, 0.65, np.nan)])
    rows = body(sb.slice_table(df, b, "severity"))
    assert rows[0][6].startswith("**0.900") and not rows[0][8].startswith("**")      # AUC: round 3 wins
    assert rows[0][9] == "**0.850**" and rows[0][7] == "0.800"                        # within-city: 2025-08 wins
    assert rows[1][8].startswith("**0.650") and rows[1][7] == "—" and rows[1][9] == "—"
    assert "| round 3 AUC [95% CI] | within-city |" in sb.slice_table(df, b, "severity").splitlines()[0]
    # per deployment: no bold when the best VLM is a partial leg (its AUCs are on other crops)
    partial = dict(aggs)
    pb = b.copy()
    pb.loc[pb.model_class != sb.PURPOSE, "full"] = False
    partial["board"] = pb
    assert "**" not in sb.render_blocks(partial)["per_deployment"]
    # the rule follows a full row only
    assert [r[1] for r in sb._with_rule([["", "a"], ["", "b"]], [False, False], 2)] == ["a", "b"]
    assert sb._with_rule([["", "a"], ["", "b"]], [True, False], 2)[1][1].startswith("*partial")
    assert [r[1] for r in sb._with_rule([["", "a"], ["", "b"], ["", "c"]], [True, False, False], 2)].count(
        "*partial coverage, not ranked against the rows above*") == 1
