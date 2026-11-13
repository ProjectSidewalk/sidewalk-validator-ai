"""Tests for analysis/target_scoreboard.py (issue #74): the vote rules, the expert-vote removal, the per-label frame
against the split variants (check B), the contrasts and their statistics, the privacy gate, determinism, and the
binding of the committed doc and JSON to the numbers computed by hand before the module existed.

Synthetic data only (cities city-a / city-b public, city-c private, as in tests/fixtures/adjudicator); no network,
no ``~/Data``."""
import doctest
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from analysis import confident_learning as cl
from analysis import target_scoreboard as ts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, "tests", "fixtures", "adjudicator", "cityparams.conf")
DOC = os.path.join(ROOT, "docs", "target-scoreboard-2026-11.md")
JSON = os.path.join(ROOT, "docs", "target-scoreboard-2026-11", "scoreboard.json")


def test_doctests():
    assert doctest.testmod(ts).failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# Vote rules
# ---------------------------------------------------------------------------------------------------------------------

# (agree, disagree) -> majority, majority2, symmetric2, margin, symmetric3, unanimous3, written out by hand
RULE_TABLE = [
    ((0, 0), ["", "", "", "", "", ""]),
    ((1, 0), ["correct", "", "", "", "", ""]),
    ((0, 1), ["incorrect", "", "", "", "", ""]),
    ((1, 1), ["", "", "", "", "", ""]),
    ((2, 0), ["correct", "correct", "correct", "", "", ""]),
    ((0, 2), ["incorrect", "incorrect", "incorrect", "incorrect", "", ""]),
    ((3, 0), ["correct", "correct", "correct", "correct", "correct", "correct"]),
    ((0, 3), ["incorrect", "incorrect", "incorrect", "incorrect", "incorrect", "incorrect"]),
    ((3, 1), ["correct", "correct", "correct", "", "", ""]),
    ((1, 3), ["incorrect", "incorrect", "incorrect", "incorrect", "", ""]),
    ((4, 1), ["correct", "correct", "correct", "correct", "correct", ""]),
    ((1, 4), ["incorrect", "incorrect", "incorrect", "incorrect", "incorrect", ""]),
    ((2, 1), ["correct", "correct", "", "", "", ""]),
]


def test_tally_verdicts_boundaries():
    a, d = zip(*[k for k, _ in RULE_TABLE])
    t = ts.tally_verdicts(list(a), list(d))
    assert list(t.columns) == list(ts.TALLY_RULES)
    for i, (key, want) in enumerate(RULE_TABLE):
        assert t.iloc[i].tolist() == want, key


def test_symmetric_derivations_must_agree(monkeypatch):
    real = cl.passes_rule

    def doctored(df, rule):
        out = real(df, rule)
        return ~out if rule == "symmetric" else out
    monkeypatch.setattr(cl, "passes_rule", doctored)
    with pytest.raises(AssertionError, match="symmetric3"):
        ts.tally_verdicts([0, 1], [3, 4])


def test_vote_class_and_pool_shares():
    pool = pd.DataFrame({"ai_vote": ["Disagree"] * 6 + ["Agree"],
                         "label_type": ["CurbRamp"] * 3 + ["Obstacle"] * 3 + ["CurbRamp"],
                         "h_agree": ["0", "1", "1", "2", "0", "4", "0"], "h_dis": ["0", "0", "1", "1", "2", "1", "5"]})
    rows = {r["label_type"]: r for r in ts.pool_shares(pool)}
    assert rows["all"]["n"] == 6  # the AI Agree row is not in the pool shares
    assert rows["all"]["classes"] == {"no decisive vote": 1, "1 vote": 1, "tie": 1, "light": 1,
                                      "margin_incorrect": 1, "margin_correct": 1}
    assert rows["all"]["rules"]["majority"] == {"cover": 4, "incorrect": 1}
    assert rows["all"]["rules"]["margin"] == {"cover": 2, "incorrect": 1}
    assert rows["CurbRamp"]["n"] == 3 and rows["Obstacle"]["rules"]["unanimous3"]["cover"] == 0


# ---------------------------------------------------------------------------------------------------------------------
# A synthetic gold table and split root
# ---------------------------------------------------------------------------------------------------------------------

def _gold():
    """Twelve decided items over two cities, two rounds and two types; city-a:1 and city-b:1 share a label id."""
    rows = [
        # uid, round, type, ai_vote, agree, dis, expert, source, train, target, flagged, stratum
        ("city-a:1", "1", "CurbRamp", "Disagree", 0, 2, "incorrect", "r1", True, "incorrect", False, "margin_incorrect"),
        ("city-b:1", "1", "CurbRamp", "Disagree", 0, 3, "correct", "r1", True, "incorrect", True, "margin_incorrect"),
        ("city-a:2", "1", "CurbRamp", "Disagree", 0, 0, "incorrect", "r1", False, "", False, "lone"),
        ("city-a:3", "1", "CurbRamp", "Disagree", 3, 0, "correct", "r1", True, "correct", False, "margin_correct"),
        ("city-a:4", "2", "CurbRamp", "Disagree", 1, 4, "incorrect", "r2_single", True, "incorrect", False,
         "margin_incorrect"),
        ("city-a:5", "2", "CurbRamp", "Disagree", 0, 2, "correct", "r2_consensus", True, "incorrect", False,
         "margin_incorrect"),
        ("city-b:6", "2", "Obstacle", "Disagree", 0, 2, "incorrect", "r2_matched", True, "incorrect", False,
         "margin_incorrect"),
        ("city-b:7", "2", "Obstacle", "Disagree", 0, 4, "incorrect", "r2_single", True, "incorrect", True,
         "margin_incorrect"),
        ("city-b:8", "2", "Obstacle", "Agree", 4, 0, "correct", "r2_single", True, "correct", False, "decoy_agree"),
        ("city-b:9", "2", "Obstacle", "Disagree", 2, 1, "correct", "r2_single", False, "", False, "light_voted"),
        ("city-a:10", "1", "Obstacle", "Agree", 1, 0, "correct", "r1", False, "", False, "decoy_agree"),
        ("city-a:11", "2", "Obstacle", "Disagree", 0, 2, "incorrect", "r2_single", False, "", False,
         "margin_incorrect"),
    ]
    g = pd.DataFrame(rows, columns=["label_uid", "round", "label_type", "ai_vote", "h_agree", "h_dis", "expert",
                                    "expert_source", "train_row", "label_class_r3", "cl_flagged", "stratum"])
    g["label_uid_schema"] = g.label_uid.str.replace("city-", "sidewalk_", regex=False)
    g["city_id"] = g.label_uid.str.rsplit(":", n=1).str[0]
    g["h_agree"], g["h_dis"], g["h_unsure"] = g.h_agree.astype(str), g.h_dis.astype(str), "0"
    g["decided"] = True
    g["severity"] = ["1", "2", "", "3", "1", "1", "2", "2", "1", "3", "", "1"]
    g["tags"] = ["narrow", "", "", "steep;narrow", "", "", "pole", "", "", "", "", ""]
    g["pano_id"] = [f"P{i}" for i in range(len(g))]
    g["group_r1"] = np.where(g["round"] == "1", np.where(g.stratum == "lone", "lone_disagree", "voted_disagree"), "")
    g["raters"] = "rater"
    return g


def _pool(g, expert2=None):
    p = pd.DataFrame({"label_uid_schema": g.label_uid_schema, "expert1": "", "expert2": ""})
    for uid, v in (expert2 or {}).items():
        p.loc[g.label_uid == uid, "expert2"] = v
    p["schema"] = p.label_uid_schema.str.rsplit(":", n=1).str[0]
    p["label_id"] = p.label_uid_schema.str.rsplit(":", n=1).str[1]
    p["ai_vote"] = g.ai_vote.to_numpy()
    p["label_type"] = g.label_type.to_numpy()
    p["h_agree"], p["h_dis"] = g.h_agree.to_numpy(), g.h_dis.to_numpy()
    return p


def _split_root(tmp_path, g):
    """``<Type>/split.csv``, ``cl/noisy_targets.csv`` and the four variants, built with confident_learning's own
    writers from the gold rows plus a few non-gold rows."""
    root = tmp_path / "splits"
    for t in ("CurbRamp", "Obstacle"):
        x = g[(g.label_type == t) & g.train_row]
        extra = pd.DataFrame({"label_uid": [f"city-a:{900 + i}" for i in range(3)], "split": ["train", "test", "val"],
                              "label_class": ["correct", "incorrect", "correct"], "human_agree": ["4", "0", "3"],
                              "human_disagree": ["0", "2", "0"], "flagged": [False, False, False]})
        base = pd.concat([pd.DataFrame({"label_uid": x.label_uid, "split": "train", "label_class": x.label_class_r3,
                                        "human_agree": x.h_agree, "human_disagree": x.h_dis,
                                        "flagged": x.cl_flagged}), extra], ignore_index=True)
        base["pano_id"], base["crop_path"] = "P", "c.webp"
        base["city"] = base.label_uid.str.rsplit(":", n=1).str[0]
        base["class_source"] = "margin"
        base["severity"], base["tags"] = "1", "[]"
        flagged = set(base.label_uid[base.flagged])
        base = base.drop(columns="flagged")
        d = root / t
        cl.write_split(base, str(d / "split.csv"))
        for rule in cl.RULES:
            cl.write_split(cl.rule_variant(base, rule), str(d / "cl" / rule / "split.csv"))
        for name, v in cl.cl_variants(base, flagged).items():
            cl.write_split(v, str(d / "cl" / name / "split.csv"))
        tr = base[base.split == "train"]
        pd.DataFrame({"label_uid": tr.label_uid, "p_incorrect": np.linspace(0.1, 0.9, len(tr)),
                      "flagged": ["1" if u in flagged else "0" for u in tr.label_uid]}).to_csv(
            d / "cl" / "noisy_targets.csv", index=False)
    return str(root)


def test_remove_expert_votes_and_per_city_keys(tmp_path):
    g = _gold()
    pool = _pool(g, {"city-b:1": "Disagree"})
    out = ts.remove_expert_votes(g, pool)
    assert out.loc[out.label_uid == "city-b:1", ["h_dis", "n_expert_votes"]].values.tolist() == [[2, 1]]
    assert out.loc[out.label_uid == "city-a:1", ["h_dis", "n_expert_votes"]].values.tolist() == [[2, 0]]
    root = _split_root(tmp_path, g)
    f = ts.build_frame(g, pool, root)
    assert f.label_uid.is_unique and len(f) == len(g)
    a1, b1 = f.set_index("label_uid").loc[["city-a:1", "city-b:1"]].itertuples()
    assert (a1.rule_symmetric3, b1.rule_symmetric3) == ("", "")  # b1 lost its third Disagree to the removal
    assert (a1.cl_flagged, b1.cl_flagged) == (False, True) and (a1.wrong, b1.wrong) == (False, True)
    k = ts.build_frame(g, pool, root, "keep").set_index("label_uid")
    assert k.loc["city-b:1", "rule_symmetric3"] == "incorrect" and k.loc["city-b:1", "n_expert_votes"] == 1


def test_frame_against_variants_and_check_b(tmp_path):
    g = _gold()
    root = _split_root(tmp_path, g)
    f = ts.build_frame(g, _pool(g), root)
    tr = f[f.train_row].set_index("label_uid")
    assert tr["touch_cl-drop"].equals(tr.cl_flagged) and tr["touch_cl-flip"].equals(tr.cl_flagged)
    # symmetric drops net-Disagree-2 incorrect targets; correct targets need net Agree > 2 (all pass here)
    assert set(tr.index[tr.touch_symmetric]) == {"city-a:1", "city-a:5", "city-b:6"}
    assert set(tr.index[~tr.oof_p_incorrect.notna()]) == set()
    tot = ts.check_b(f, root)
    assert tot["cl-drop"] == {"n_rated_train": 8, "target_matches_expert": 6, "touched": 2, "touched_overturned": 1,
                              "untouched": 6, "untouched_overturned": 1}


def test_masked_only_drops_exactly_the_meeting_items(tmp_path):
    g = _gold()
    root = _split_root(tmp_path, g)
    f = ts.build_frame(g, _pool(g), root)
    assert set(f.label_uid[~f.masked]) == {"city-a:5"}
    assert set(g.label_uid[g.expert_source == ts.CONSENSUS]) == {"city-a:5"}


def test_contrasts_by_round(tmp_path):
    g = _gold()
    f = ts.build_frame(g, _pool(g), _split_root(tmp_path, g))
    p1 = ts.p1_rows(f, n_boot=0)
    # margin-incorrect AI Disagrees: a1 (0,2) b1 (0,3) a4 (1,4) a5 (0,2) b6 (0,2) b7 (0,4) a11 (0,2)
    assert (p1["pooled"]["a"]["n"], p1["pooled"]["b"]["n"]) == (3, 4)
    assert (p1["pooled"]["a"]["k"], p1["pooled"]["b"]["k"]) == (2, 3)
    assert p1["1"]["a"]["n"] + p1["2"]["a"]["n"] == 3
    assert p1["pooled"]["lo"] is None  # groups under MIN_CELL: no interval
    assert p1["heterogeneity"]["did"] is None
    p2 = ts.p2_rows(f, "cl-drop", n_boot=0)
    assert (p2["pooled"]["a"]["k"], p2["pooled"]["a"]["n"], p2["pooled"]["b"]["k"], p2["pooled"]["b"]["n"]) == \
        (1, 2, 1, 6)
    assert p2["pooled"]["recall"]["k"] == 1 and p2["pooled"]["recall"]["n"] == 2


def test_rule_rows_and_majority_collapse(tmp_path):
    g = _gold()
    f = ts.build_frame(g, _pool(g), _split_root(tmp_path, g))
    r = ts.rule_rows(f)
    assert r["majority_equals_majority2"] is True  # no single-vote AI Disagree in this sample, as in the real one
    m = r["rows"]["pooled"]["margin"]
    assert (m["incorrect"]["k"], m["incorrect"]["n"]) == (5, 7)
    d = ts.rule_rows(f, "decoys")
    assert d["majority_equals_majority2"] is False  # the (1, 0) decoy has a majority but not two votes


def test_fisher_reference_values():
    # SciPy 1.x fisher_exact, computed outside the repo (CI does not install SciPy)
    assert ts.fisher_exact(4, 9, 19, 99) == pytest.approx(0.2424698856612656, rel=1e-9)
    assert ts.fisher_exact(3, 2, 17, 83) == pytest.approx(0.046573103912332964, rel=1e-9)
    assert ts.fisher_exact(10, 0, 0, 10) == pytest.approx(1.082508822446903e-05, rel=1e-9)
    assert ts.fisher_exact(5, 5, 5, 5) == 1.0


def test_diff_boot_is_seeded():
    ok = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    a = [True] * 6 + [False] * 6
    assert ts.diff_boot(ok, a, 500, 3) == ts.diff_boot(ok, a, 500, 3)
    assert ts.diff_boot(ok, a, 500, 3)[1:] != ts.diff_boot(ok, a, 500, 4)[1:]


def test_poststratified_excludes_small_cells():
    x = pd.DataFrame({"t": ["A"] * 6 + ["B"] * 4, "ok": [True] * 6 + [False] * 4})
    r = ts.poststratified(x, ["t"], {("A",): 10, ("B",): 1000}, pd.Series(True, index=x.index), x.ok)
    assert r["excluded"] == ["B(n=4)"] and r["rate"] == pytest.approx(1.0)


def test_poststratified_weights_by_population_times_domain_share():
    # cell A: 10 items, 6 in the domain, 3/6 succeed; cell B: 10 items, 5 in the domain, 5/5 succeed. Both cells have
    # population 100, so the domain weights are 100 * 6/10 = 60 and 100 * 5/10 = 50: (60 * 0.5 + 50 * 1) / 110.
    x = pd.DataFrame({"t": ["A"] * 10 + ["B"] * 10,
                      "dom": [True] * 6 + [False] * 4 + [True] * 5 + [False] * 5,
                      "ok": [True] * 3 + [False] * 7 + [True] * 5 + [False] * 5})
    r = ts.poststratified(x, ["t"], {("A",): 100, ("B",): 100}, x.dom, x.ok)
    assert r["rate"] == pytest.approx(80 / 110)
    assert r["rate"] != pytest.approx(0.75)  # what equal cell weights (ignoring the domain share) would give
    assert r["n"] == 11 and r["excluded"] == []


def _ac(k, n):
    p = (k + 2) / (n + 4)
    return p * (1 - p) / n


def _p1_frame():
    """Round 1: kept rows fill only CurbRamp and dropped rows only Obstacle (no shared cell). Round 2: one type in
    two cities, both groups >= 5 in each (two shared cells); city-a's kept cell is 5/5 (a rate at 100%)."""
    rows = []

    def add(rnd, city, t, kept, ok, n):
        for i in range(n):
            rows.append({"round": rnd, "ai_vote": "Disagree", "city_id": city, "label_type": t,
                         "group_r1": "voted_disagree" if rnd == "1" else "", "stratum": "margin_incorrect",
                         "rule_margin": "incorrect", "rule_symmetric3": "incorrect" if kept else "",
                         "expert": "incorrect" if i < ok else "correct"})
    add("1", "city-a", "CurbRamp", True, 4, 5)
    add("1", "city-a", "CurbRamp", False, 1, 1)
    add("1", "city-a", "Obstacle", True, 1, 1)
    add("1", "city-a", "Obstacle", False, 3, 5)
    add("2", "city-a", "NoCurbRamp", True, 5, 5)
    add("2", "city-a", "NoCurbRamp", False, 3, 5)
    add("2", "city-b", "NoCurbRamp", True, 3, 6)
    add("2", "city-b", "NoCurbRamp", False, 3, 6)
    return pd.DataFrame(rows)


def test_weighted_p1_uses_shared_cells_and_an_unclipped_interval():
    w = {"1": {("voted_disagree", t): 100 for t in ts.LABEL_TYPES},
         "2": {(c, "margin_incorrect", "NoCurbRamp"): 100 for c in ("city-a", "city-b")}}
    out = ts.weighted_p1(_p1_frame(), w)
    s1 = out["1"]["shared"]
    assert (s1["cells"], s1["cells_a"], s1["cells_b"], s1["cells_shared"]) == (2, 1, 1, 0)
    assert s1["estimable"] is False and s1["diff"] is None and s1["lo"] is None
    s2 = out["2"]["shared"]
    assert s2["estimable"] is True and s2["cells_shared"] == 2 and (s2["n_a"], s2["n_b"]) == (11, 11)
    # equal weights: kept (1.0 + 0.5) / 2 = 0.75, dropped (0.6 + 0.5) / 2 = 0.55, kept minus dropped +0.2
    assert (s2["rate_a"], s2["rate_b"]) == (pytest.approx(0.75), pytest.approx(0.55))
    assert s2["diff"] == pytest.approx(0.2)
    se = 0.5 * np.sqrt(_ac(5, 5) + _ac(3, 5) + _ac(3, 6) + _ac(3, 6))
    assert s2["se"] == pytest.approx(se)
    assert (s2["lo"], s2["hi"]) == (pytest.approx(0.2 - 1.96 * se), pytest.approx(0.2 + 1.96 * se))
    assert s2["weight_share"] == pytest.approx(1.0)
    assert out["2"]["all"]["rate"] == pytest.approx(0.65)


def test_shared_cell_diff_one_shared_cell_is_not_estimable():
    f = _p1_frame()
    f = f[f["round"] == "2"]
    f = f[~((f.city_id == "city-b") & (f.rule_symmetric3 == ""))].iloc[:-1]  # city-b keeps 5 kept rows, no dropped
    r = ts.shared_cell_diff(f, ["city_id"], {("city-a",): 100, ("city-b",): 100}, pd.Series(True, index=f.index),
                            f.rule_symmetric3 == "incorrect", f.expert == "incorrect")
    assert (r["cells"], r["cells_a"], r["cells_b"], r["cells_shared"]) == (2, 2, 1, 1)
    assert r["estimable"] is False and r["diff"] is None
    assert 0 < r["weight_share"] < 1
    text = ts.render_weighted({"weighted": {k: {"all": {"rate": None, "excluded": []}, "shared": r}
                                            for k in ("1", "2")}})
    assert "not estimable: 1 shared cell (needs 2)" in text


def test_gain_boot_is_kept_minus_all():
    ok = [1, 1, 1, 0, 1, 0, 0, 0]
    a = [True] * 4 + [False] * 4
    g, lo, hi = ts.gain_boot(ok, a, 500, 1)
    assert g == pytest.approx(0.75 - 0.5)
    d = ts.diff_boot(ok, a, 0)[0]
    assert g == pytest.approx(d * 4 / 8)  # the A-minus-B difference times B's share
    assert lo <= g <= hi and ts.gain_boot(ok, a, 500, 1) == (g, lo, hi)


def test_did_boot_numeric_path():
    # round 1: A 5/6 vs B 1/6 (+4/6); round 2: A 3/6 vs B 3/6 (0)
    ok = [1, 1, 1, 1, 1, 0] + [1, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0, 0] * 2
    a = ([True] * 6 + [False] * 6) * 2
    grp = ["1"] * 12 + ["2"] * 12
    d, lo, hi = ts.did_boot(ok, a, grp, 1000, 5)
    assert d == pytest.approx(4 / 6)
    assert lo < d < hi and lo > -1 and hi < 2
    assert ts.did_boot(ok, a, grp, 1000, 5) == (d, lo, hi)
    assert ts.did_boot(ok, [not x for x in a], grp, 1000, 5)[0] == pytest.approx(-4 / 6)
    assert ts.did_boot(ok, a, ["2"] * 12 + ["1"] * 12, 1000, 5)[0] == pytest.approx(-4 / 6)  # level order matters


def test_slice_rows_blanks_thin_cells():
    g = pd.DataFrame({"label_uid": [f"city-a:{i}" for i in range(7)], "ai_vote": "Disagree",
                      "label_type": "CurbRamp", "severity": ["1"] * 4 + ["2"] * 2 + ["3"],
                      "tags": ["narrow"] * 4 + [""] * 3, "expert": ["incorrect"] * 3 + ["correct"] * 4,
                      "rule_margin": "incorrect", "rule_symmetric3": ["incorrect"] * 3 + [""] * 4})
    train = pd.DataFrame({"label_type": ["CurbRamp"] * 3, "severity": ["1", "1", "2"], "tags": ["[]"] * 3})
    s = ts.slice_rows(g, train, min_n=3)
    sev = {r["slice"]: r for r in s["ai_precision_severity"]}
    assert sev["1"]["rate"] == pytest.approx(0.75) and sev["1"]["train"] == 2
    assert sev["2"]["rate"] is None and (sev["2"]["k"], sev["2"]["n"]) == (0, 2) and sev["2"]["lo"] is None
    tag = {r["slice"]: r for r in s["ai_precision_tag"]}
    assert tag["narrow"]["rate"] == pytest.approx(0.75) and tag["(none)"]["rate"] == pytest.approx(0.0)
    p1 = {r["slice"]: r for r in s["p1_severity"]}
    assert p1["1"]["kept"]["rate"] == pytest.approx(1.0) and p1["1"]["dropped"]["rate"] is None
    assert p1["1"]["diff"] is None and p1["1"]["fisher_p"] is None
    text = ts.render_slices({"slices": s})
    assert "| 1 | 3/3 (100.0%" in text and "n 1 |" in text  # a thin group shows its size, not k/n


def test_ranking_rows_direction():
    # train rows: wrong `incorrect` targets have a low p_incorrect, wrong `correct` targets a high one;
    # AI Disagrees: wrong margin targets have the smaller lead. Every score separates perfectly, so AUC is 1.
    inc_p = [0.1, 0.2, 0.3] + [0.6, 0.7, 0.8, 0.9]
    cor_p = [0.9, 0.8] + [0.1, 0.2]
    tr = pd.DataFrame({"train_row": True, "ai_vote": "Agree", "rule_margin": "",
                       "target": ["incorrect"] * 7 + ["correct"] * 4, "oof_p_incorrect": inc_p + cor_p,
                       "wrong": [True] * 3 + [False] * 4 + [True] * 2 + [False] * 2, "h_agree": 0, "h_dis": 0,
                       "expert": "", "pano_id": [f"T{i}" for i in range(11)]})
    dis = pd.DataFrame({"train_row": False, "ai_vote": "Disagree", "target": "", "oof_p_incorrect": np.nan,
                        "wrong": False,
                        "rule_margin": ["incorrect"] * 5 + ["correct"] * 4,
                        "h_agree": [0, 0, 0, 0, 0] + [3, 3, 6, 6], "h_dis": [2, 2, 5, 5, 5] + [0, 0, 0, 0],
                        "expert": ["correct"] * 2 + ["incorrect"] * 3 + ["incorrect"] * 2 + ["correct"] * 2,
                        "pano_id": [f"D{i}" for i in range(9)]})
    rows = ts.ranking_rows(pd.concat([tr, dis], ignore_index=True), n_boot=0)
    got = {(r["score"].split()[0], r["target_class"]): r for r in rows}
    for key, n, nw in ((("oof", "incorrect"), 7, 3), (("oof", "correct"), 4, 2), (("vote", "incorrect"), 5, 2),
                       (("vote", "correct"), 4, 2)):
        assert (got[key]["n"], got[key]["n_wrong"]) == (n, nw), key
        assert got[key]["auc"] == pytest.approx(1.0), key
        assert got[key]["lo"] is None


def test_build_frame_refuses_a_wrong_split_root_and_no_flags(tmp_path):
    g = _gold()
    with pytest.raises(FileNotFoundError, match="--split-root"):
        ts.build_frame(g, _pool(g), str(tmp_path / "nowhere"))
    root = _split_root(tmp_path, g)
    with pytest.raises(FileNotFoundError, match="--split-root"):
        ts.check_b(ts.build_frame(g, _pool(g), root), str(tmp_path / "nowhere"))
    g2 = g.assign(cl_flagged=False)
    with pytest.raises(ValueError, match="flagged"):
        ts.build_frame(g2, _pool(g2), root)


def test_score_cli_names_a_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "city_lists", lambda: LISTS)
    args = _cli_inputs(tmp_path, _gold())
    args[args.index("--gold") + 1] = str(tmp_path / "no-gold.csv")
    with pytest.raises(SystemExit, match="input not found: --gold"):
        ts.main(args + ["--out-dir", str(tmp_path / "p"), "--out", str(tmp_path / "pub")])


def _weights():
    return {"1": {("voted_disagree", t): 100 for t in ts.LABEL_TYPES},
            "2": {(c, s, t): 50 for c in ("city-a", "city-b") for s in ("margin_incorrect",) for t in ts.LABEL_TYPES}}


def test_score_is_deterministic_and_counts_only(tmp_path):
    g = _gold()
    root = _split_root(tmp_path, g)
    j1, _ = ts.score(g, _pool(g), root, _weights(), n_boot=200, seed=0)
    j2, _ = ts.score(g, _pool(g), root, _weights(), n_boot=200, seed=0)
    assert ts.dumps(j1) == ts.dumps(j2)
    j3, _ = ts.score(g, _pool(g), root, _weights(), n_boot=200, seed=1)
    strip = re.compile(r'"(lo|hi|did|seed)": [-0-9.e]+')
    assert strip.sub("", ts.dumps(j1)) == strip.sub("", ts.dumps(j3))  # only interval fields change with the seed
    text = ts.dumps(j1)
    for key in ts.FORBIDDEN_KEYS:
        assert f'"{key}"' not in text
    assert "city-a:" not in text and "rater" not in text
    blocks = ts.render_blocks({**j1, "provenance": _prov()})
    assert set(blocks) == set(ts.BLOCKS)


def _prov():
    return {"gold_sha256": "x", "gold_matches_pin": False, "pool_sha256": "x", "r1_manifest_sha256": "x",
            "r2_recount_sha256": "x", "r2_design_sha256": "x", "splits_combined_sha256": "x", "n_split_files": 0,
            "verdicts_sha256": "x", "git_commit": "x", "python": "3", "pandas": "2", "numpy": "1", "n_boot": 0,
            "seed": 0, "cities_read_at": "t", "n_public": 2}


def test_round_weights_never_carry_rater_names():
    w = ts.round_weights({"population": {"voted_disagree": {"CurbRamp": 7}}},
                         {"after": {"population": {"rater_x": {"margin_incorrect": {"CurbRamp": 9}}}}},
                         {"raters": {"rater_x": "sidewalk_b"}}, {"sidewalk_b": "city-b"})
    assert w == {"1": {("voted_disagree", "CurbRamp"): 7}, "2": {("city-b", "margin_incorrect", "CurbRamp"): 9}}


# ---------------------------------------------------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------------------------------------------------

LISTS = {"public": {"city-a", "city-b"}, "non_public_terms": {"city-c", "Hidden Town"}, "read_at": "t",
         "source": "test"}
ALL = {"city-a": "sidewalk_a", "city-b": "sidewalk_b", "city-c": "sidewalk_c"}


def test_privacy_gate():
    ts.privacy_gate("city-a and city-b", LISTS, ALL)
    for bad in ("a city-c row", "from sidewalk_c", "HIDDEN TOWN", '{"label_uid": 1}', '{"raters": "x"}'):
        with pytest.raises(ValueError):
            ts.privacy_gate(bad, LISTS, ALL)


def _cli_inputs(tmp_path, g, private_r2_city=False):
    root = _split_root(tmp_path, g)
    gp = tmp_path / "gold.csv"
    g.to_csv(gp, index=False)
    pp = tmp_path / "pool.csv"
    _pool(g).drop(columns="label_uid_schema").to_csv(pp, index=False)
    r1 = tmp_path / "r1.json"
    r1.write_text(json.dumps({"population": {"voted_disagree": {t: 100 for t in ts.LABEL_TYPES}}}))
    r2 = tmp_path / "r2.json"
    r2.write_text(json.dumps({"after": {"population": {"r": {"margin_incorrect": {t: 50 for t in ts.LABEL_TYPES}}}}}))
    des = tmp_path / "design.json"
    des.write_text(json.dumps({"raters": {"r": "sidewalk_c" if private_r2_city else "sidewalk_b"}}))
    return ["score", "--gold", str(gp), "--pool", str(pp), "--split-root", root, "--r1-manifest", str(r1),
            "--r2-recount", str(r2), "--r2-design", str(des), "--cityparams", CONF, "--n-boot", "50"]


def test_score_cli_writes_and_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "city_lists", lambda: LISTS)
    g = _gold()
    args = _cli_inputs(tmp_path, g)
    with pytest.raises(SystemExit, match="inside a git checkout"):
        ts.main(args + ["--out-dir", os.path.join(ROOT, "tmp-should-not-exist"), "--out", str(tmp_path / "pub")])
    priv = tmp_path / "private"
    ts.main(args + ["--out-dir", str(priv), "--out", str(tmp_path / "pub")])
    assert (priv / "verdicts.csv").exists() and (priv / "run.json").exists()
    assert (tmp_path / "pub" / "scoreboard.json").exists()
    v = pd.read_csv(priv / "verdicts.csv", dtype=str)
    assert "raters" not in v.columns and v.label_uid.is_unique
    # a private deployment reaching the public text (here through round 2's weight cells) is refused
    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    g2 = _gold()
    g2.loc[g2["round"] == "2", "city_id"] = "city-c"
    with pytest.raises(SystemExit, match="refusing"):
        ts.main(_cli_inputs(tmp2, g2, private_r2_city=True) + ["--out-dir", str(tmp2 / "p"), "--out",
                                                              str(tmp2 / "pub")])
    assert not (tmp2 / "pub" / "scoreboard.json").exists()


# ---------------------------------------------------------------------------------------------------------------------
# The committed doc and JSON
# ---------------------------------------------------------------------------------------------------------------------

def _committed():
    if not os.path.exists(JSON):
        pytest.fail("docs/target-scoreboard-2026-11/scoreboard.json is missing")
    return json.load(open(JSON)), open(DOC).read()


def test_doc_matches_json():
    j, doc = _committed()
    assert ts.splice(doc, ts.render_blocks(j)) == doc


def test_committed_json_reproduces_the_hand_numbers():
    """The plan (Fable 5.2) and its review (Opus 5.5) computed these by hand from the same gold table before the
    module existed; the run must reproduce every one."""
    j, _ = _committed()
    c = j["crosscheck"]
    assert j["provenance"]["gold_sha256"] == ts.GOLD_SHA256 and j["provenance"]["gold_matches_pin"] is True
    assert j["gold_counts"]["n_items"] == 800
    assert (c["decided"], c["ai_disagree_decided"], c["train_rows"], c["flagged"]) == (741, 497, 131, 13)
    r = c["rules"]
    assert (r["margin"]["cover"], r["symmetric3"]["cover"], r["unanimous3"]["cover"]) == (219, 133, 99)
    assert [round(100 * r[x]["incorrect_right"] / r[x]["incorrect"], 1) for x in ("margin", "symmetric3",
                                                                                   "unanimous3")] == [81.6, 83.3, 81.8]
    assert (r["majority"]["cover"], round(100 * r["majority"]["agree"] / r["majority"]["cover"], 1)) == (303, 75.2)
    assert c["majority_equals_majority2"] is True
    assert (c["flagged_wrong"], c["flagged"], c["unflagged_wrong"], c["unflagged"]) == (4, 13, 19, 118)
    assert {k: round(100 * v, 1) for k, v in c["pool_cover_share"].items() if k != "symmetric2"} == {
        "majority": 59.5, "majority2": 31.5, "margin": 19.8, "symmetric3": 11.7, "unanimous3": 6.4}
    assert j["check_b"]["cl-drop"]["touched_overturned"] == 4


def test_doc_prose_quotes_the_json():
    j, doc = _committed()
    prose = re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED: \w+ -->", "", doc, flags=re.S)
    prose = " ".join(prose.split())
    p1, p2 = j["primary"]["P1"]["pooled"], j["primary"]["P2"]["pooled"]
    for r in (p1, p2):
        assert f"{r['a']['k']}/{r['a']['n']}" in prose and f"{r['b']['k']}/{r['b']['n']}" in prose
        assert f"{100 * r['diff']:+.1f}" in prose and f"{100 * r['lo']:+.1f}" in prose and f"{100 * r['hi']:+.1f}" in prose
    assert f"{100 * p1['b']['lo']:.1f}%" in prose  # the dropped-row lower bound the reading rests on
    gain = p1["gain"]
    assert f"{100 * gain['gain']:+.1f} pts [{100 * gain['lo']:+.1f}, {100 * gain['hi']:+.1f}]" in prose
    assert f"[{100 * p2['a']['lo']:.1f}, {100 * p2['a']['hi']:.1f}]" in prose  # the flagged-wrong share's interval
    assert j["primary"]["roles"] == {"P1": "primary", "P2": "primary", "P1_unanimous3": "descriptive"}
    assert "Weighting does not change the reading" not in prose
    assert all(not j["weighted"][r]["shared"]["estimable"] or j["weighted"][r]["shared"]["diff"] is not None
               for r in ("1", "2"))
    assert str(j["gold_counts"]["n_ai_disagree_masked"]) in prose
    assert not re.search(r"\bbl[i]nd", doc, re.I)  # the house word is "masked"


def test_committed_json_is_counts_only():
    j, _ = _committed()
    text = json.dumps(j)
    for key in ts.FORBIDDEN_KEYS:
        assert f'"{key}"' not in text
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", text)  # no user ids
    assert not re.search(r"\b[a-z-]+:\d+\b", text)  # no city:label_id
