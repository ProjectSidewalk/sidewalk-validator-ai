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


def _splits(n_per=60):
    rows = []
    for t in sb.LABEL_TYPES:
        for i in range(n_per):
            c = CITIES[i % len(CITIES)]
            rows.append({"label_uid": f"{c}:{sb.LABEL_TYPES.index(t)}{i:04d}", "city": c, "label_type": t,
                         "label_class": "incorrect" if (i // len(CITIES)) % 2 else "correct",
                         "split": "test_heldout" if c in ("tucson-az", *PRIVATE_HELDOUT) else "test"})
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
                         ("short_answer", "board", "per_type", "paired", "per_deployment", "timing", "coverage"))
    path.write_text(f"# Test board\n\n{prose}\n\n{blocks}\n")


def _prose(aggs):
    f = sb.facts(aggs)
    return (f"The board has {f['legs']} legs, {f['full_legs']} of them scored all {f['n_total']:,} held-out crops. "
            f"Round 3 leads the best VLM ({f['best_vlm']}) by {f['gap']:.3f} macro-AUC on the crops both scored.")


@pytest.fixture
def board(tmp_path):
    stage, out, doc = tmp_path / "stage", tmp_path / "out", tmp_path / "doc.md"
    _stage(stage)
    pub = tmp_path / "public.txt"
    pub.write_text("\n".join(PUBLIC))
    base = ["build", "--scores-dir", str(stage), "--out-dir", str(out), "--doc", str(doc), "--public-cities",
            str(pub), "--n-boot", "50"]
    _doc(doc)
    sb.main(base)                                   # aggregates, figures (prose not yet written)
    _doc(doc, _prose(sb.read_aggregates(str(out))))
    sb.main(base)                                   # fills the blocks
    return {"stage": stage, "out": out, "doc": doc, "base": base, "pub": pub}


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
    for f in ("leaderboard", "by_type", "roc", "per_deployment", "latency"):
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
