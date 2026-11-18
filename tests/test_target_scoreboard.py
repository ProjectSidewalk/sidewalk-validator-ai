"""Tests for analysis/target_scoreboard.py (issue #74): the vote rules, the expert-vote removal, the per-label frame
against the split variants (check B), the contrasts and their statistics, the privacy gate, determinism, and the
binding of the committed doc and JSON to the numbers computed by hand before the module existed.

Synthetic data only (cities city-a / city-b public, city-c private, as in tests/fixtures/adjudicator); no network,
no ``~/Data``."""
import doctest
import itertools
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
    assert (m["incorrect"]["k"], m["incorrect"]["n"]) == (None, 7)  # round 1's cell has 2 items, so pooled hides
    assert (r["rows"]["2"]["margin"]["incorrect"]["k"], r["rows"]["2"]["margin"]["incorrect"]["n"]) == (4, 5)
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


def test_shared_cell_diff_renormalises_over_shared_cells_and_weights_by_domain_share():
    # cell A: 20 items, 10 in the domain (5 kept 5/5 incorrect, 5 dropped 2/5); cell B: 10 items, all in the domain
    # (5 kept 3/5, 5 dropped 3/5); cell C: kept only, so not shared but it carries weight. Population 100 each, so the
    # domain weights are A 100 * 10/20 = 50, B 100 * 10/10 = 100, renormalised over A and B: 1/3 and 2/3.
    a_kept = [True] * 5 + [False] * 5 + [False] * 10
    a_dom = [True] * 10 + [False] * 10
    a_ok = [True] * 5 + [True, True, False, False, False] + [False] * 10
    b_kept = [True] * 5 + [False] * 5
    b_ok = [True, True, True, False, False] * 2
    x = pd.DataFrame({"t": ["A"] * 20 + ["B"] * 10 + ["C"] * 6,
                      "kept": a_kept + b_kept + [True] * 6,
                      "dom": a_dom + [True] * 10 + [True] * 6,
                      "ok": a_ok + b_ok + [True] * 6})
    r = ts.shared_cell_diff(x, ["t"], {("A",): 100, ("B",): 100, ("C",): 100}, x.dom, x.kept, x.ok)
    assert (r["cells"], r["cells_a"], r["cells_b"], r["cells_shared"]) == (3, 3, 2, 2)
    assert r["estimable"] is True and (r["n_a"], r["n_b"]) == (10, 10)
    assert r["rate_a"] == pytest.approx(1 / 3 * 1.0 + 2 / 3 * 0.6)
    assert r["rate_b"] == pytest.approx(1 / 3 * 0.4 + 2 / 3 * 0.6)
    assert r["diff"] == pytest.approx(0.2)
    assert r["diff"] != pytest.approx(0.3)  # what equal cell weights (no domain share) would give
    assert r["weight_share"] == pytest.approx(150 / 250)
    se = np.sqrt((1 / 3) ** 2 * (_ac(5, 5) + _ac(2, 5)) + (2 / 3) ** 2 * (_ac(3, 5) + _ac(3, 5)))
    assert r["se"] == pytest.approx(se)


def test_weighted_p1_shared_diff_reads_only_margin_incorrect_rows():
    w = {"1": {("voted_disagree", t): 100 for t in ts.LABEL_TYPES},
         "2": {(c, "margin_incorrect", "NoCurbRamp"): 100 for c in ("city-a", "city-b")}}
    f = _p1_frame()
    extra = pd.DataFrame([{"round": "2", "ai_vote": "Disagree", "city_id": c, "label_type": "NoCurbRamp",
                           "group_r1": "", "stratum": "margin_incorrect", "rule_margin": "correct",
                           "rule_symmetric3": "", "expert": "correct"}
                          for c, n in (("city-a", 5), ("city-b", 6)) for _ in range(n)])
    s2 = ts.weighted_p1(pd.concat([f, extra], ignore_index=True), w)["2"]["shared"]
    # the margin-correct rows cut each cell's domain share to 2/3 (10 of 15, 12 of 18), so the cells keep equal
    # weights and the result equals the fixture's own
    assert (s2["n_a"], s2["n_b"]) == (11, 11)
    assert s2["diff"] == pytest.approx(0.2)


def test_render_weighted_labels_kept_and_dropped():
    s = ts.weighted_p1(_p1_frame(), {"1": {("voted_disagree", t): 100 for t in ts.LABEL_TYPES},
                                     "2": {(c, "margin_incorrect", "NoCurbRamp"): 100
                                           for c in ("city-a", "city-b")}})
    text = ts.render_weighted({"weighted": s})
    assert "kept 75.0% (n 11), dropped 55.0% (n 11)" in text


def test_gain_boot_is_kept_minus_all():
    ok = [1, 1, 1, 0, 1, 0, 0, 0]
    a = [True] * 4 + [False] * 4
    g, lo, hi = ts.gain_boot(ok, a, 500, 1)
    assert g == pytest.approx(0.75 - 0.5)
    d = ts.diff_boot(ok, a, 0)[0]
    assert g == pytest.approx(d * 4 / 8)  # the A-minus-B difference times B's share
    assert lo <= g <= hi and ts.gain_boot(ok, a, 500, 1) == (g, lo, hi)


def _p1_rows_frame():
    """Round 1: kept 6 (5 incorrect) against dropped 6 (2 incorrect), both groups at MIN_CELL or more. Round 2: kept
    2 (2 incorrect) against dropped 8 (3 incorrect), kept under MIN_CELL. Plus rows p1_rows must ignore: an AI Agree
    and a margin-``correct`` AI Disagree."""
    rows = []

    def add(rnd, kept, ok, n, ai="Disagree", margin="incorrect"):
        for i in range(n):
            rows.append({"round": rnd, "ai_vote": ai, "rule_margin": margin,
                         "rule_symmetric3": "incorrect" if kept else "", "rule_unanimous3": "",
                         "expert": "incorrect" if i < ok else "correct"})
    add("1", True, 5, 6)
    add("1", False, 2, 6)
    add("2", True, 2, 2)
    add("2", False, 3, 8)
    add("2", False, 0, 3, ai="Agree")
    add("2", False, 0, 3, margin="correct")
    return pd.DataFrame(rows)


def test_p1_rows_gain_is_kept_minus_all_with_its_own_interval():
    p1 = ts.p1_rows(_p1_rows_frame(), n_boot=2000, seed=3)
    r1, r2, pooled = p1["1"], p1["2"], p1["pooled"]
    # round 1: kept 5/6, all 7/12; round 2: kept 2/2, all 5/10; pooled: kept 7/8, all 12/22
    assert r1["gain"]["gain"] == pytest.approx(5 / 6 - 7 / 12)
    assert r2["gain"]["gain"] == pytest.approx(1.0 - 0.5)
    assert pooled["gain"]["gain"] == pytest.approx(7 / 8 - 12 / 22)
    assert pooled["gain"]["gain"] == pytest.approx(pooled["diff"] * 14 / 22)  # kept minus dropped times B's share
    # the gain's interval is its own bootstrap, about B's share of the kept-minus-dropped interval, not that interval
    for r in (r1, pooled):
        g = r["gain"]
        assert g["lo"] <= g["gain"] <= g["hi"]
        assert (g["hi"] - g["lo"]) < 0.85 * (r["hi"] - r["lo"])  # about 0.5 (round 1) and 0.7 (pooled) here
    # a round with a group under MIN_CELL has a point but no interval, like its kept-minus-dropped row
    assert r2["gain"]["gain"] is not None and r2["gain"]["lo"] is None and r2["gain"]["hi"] is None
    assert r2["lo"] is None


def test_contrast_blanks_the_interval_under_min_cell():
    ok = [1, 1, 0] + [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    thin = ts.contrast(ok, [True] * 3 + [False] * 10, n_boot=500, seed=1)
    assert thin["diff"] == pytest.approx(2 / 3 - 0.2) and thin["lo"] is None and thin["hi"] is None
    ok5 = [1, 1, 0, 1, 1] + [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    full = ts.contrast(ok5, [True] * 5 + [False] * 10, n_boot=500, seed=1)
    assert full["lo"] is not None and full["lo"] <= full["diff"] <= full["hi"]


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
    assert sev["2"]["rate"] is None and (sev["2"]["k"], sev["2"]["n"]) == (None, 2) and sev["2"]["lo"] is None
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


def test_split_files_needed_and_a_missing_noisy_targets_file(tmp_path):
    need = ts.split_files_needed("R", "CurbRamp")
    assert need == [os.path.join("R", "CurbRamp", rel) for rel in
                    ["split.csv", "cl/noisy_targets.csv"] + [f"cl/{v}/split.csv" for v in ts.VARIANTS]]
    g = _gold()
    root = _split_root(tmp_path, g)
    os.remove(os.path.join(root, "Obstacle", "cl", "noisy_targets.csv"))
    with pytest.raises(FileNotFoundError, match="noisy_targets"):
        ts.build_frame(g, _pool(g), root)


def test_score_cli_names_a_missing_split_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "city_lists", lambda: LISTS)
    args = _cli_inputs(tmp_path, _gold())
    args[args.index("--split-root") + 1] = str(tmp_path / "no-splits")
    with pytest.raises(SystemExit, match="input not found: --split-root"):
        ts.main(args + ["--out-dir", str(tmp_path / "p"), "--out", str(tmp_path / "pub")])


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


def test_vote_class_boundaries():
    a = [0, 0, 1, 2, 3, 1, 1, 0, 5]
    d = [0, 1, 1, 0, 0, 2, 3, 2, 2]
    assert ts.vote_class(a, d).tolist() == ["no decisive vote", "1 vote", "tie", "light", "margin_correct", "light",
                                            "margin_incorrect", "margin_incorrect", "margin_correct"]


def test_contrast_fisher_p_is_of_the_a_b_table():
    # A: 3 of 5 succeed, B: 17 of 100 (SciPy 0.046573...); a mis-ordered table gives another p
    ok = [1] * 3 + [0] * 2 + [1] * 17 + [0] * 83
    c = ts.contrast(ok, [True] * 5 + [False] * 100, n_boot=0)
    assert c["fisher_p"] == pytest.approx(0.046573103912332964, rel=1e-9)
    assert (c["a"]["k"], c["a"]["n"], c["b"]["k"], c["b"]["n"]) == (3, 5, 17, 100)
    c2 = ts.contrast([1] * 4 + [0] * 9 + [1] * 19 + [0] * 99, [True] * 13 + [False] * 118, n_boot=0)
    assert c2["fisher_p"] == pytest.approx(0.2424698856612656, rel=1e-9)


def test_p2_rows_denominators(tmp_path):
    g = _gold()
    f = ts.build_frame(g, _pool(g), _split_root(tmp_path, g))
    r = ts.p2_rows(f, "cl-drop", n_boot=0)["pooled"]
    # 8 train rows: flagged b1 (wrong) and b7 (right); unflagged 6, of which a5 is wrong
    assert (r["a"]["k"], r["a"]["n"], r["b"]["k"], r["b"]["n"]) == (1, 2, 1, 6)
    assert (r["recall"]["k"], r["recall"]["n"]) == (1, 2)
    assert (r["false_alarm"]["k"], r["false_alarm"]["n"]) == (1, 6)
    assert (r["kept_accuracy"]["k"], r["kept_accuracy"]["n"]) == (5, 6)
    # symmetric touches a1, a5 (wrong) and b6: the touched and the right-target counts differ, so each denominator
    # is pinned to its own group
    sy = ts.p2_rows(f, "symmetric", n_boot=0)["pooled"]
    assert (sy["a"]["k"], sy["a"]["n"], sy["b"]["k"], sy["b"]["n"]) == (1, 3, 1, 5)
    assert (sy["recall"]["k"], sy["recall"]["n"]) == (1, 2)
    assert (sy["false_alarm"]["k"], sy["false_alarm"]["n"]) == (2, 6)
    assert (sy["kept_accuracy"]["k"], sy["kept_accuracy"]["n"]) == (4, 5)
    inc = ts.p2_rows(f, "cl-drop", "incorrect", n_boot=0)["pooled"]
    assert inc["a"]["n"] + inc["b"]["n"] == int((f.train_row & (f.target == "incorrect")).sum()) == 6


def test_score_primaries_are_masked_only(tmp_path):
    g = _gold()
    j, frame = ts.score(g, _pool(g), _split_root(tmp_path, g), _weights(), n_boot=0, seed=0)
    p1 = j["primary"]["P1"]["pooled"]
    masked = ts.p1_rows(frame[frame.masked], n_boot=0)["pooled"]
    assert (p1["a"]["n"], p1["b"]["n"]) == (masked["a"]["n"], masked["b"]["n"]) == (3, 3)  # city-a:5 left out
    full = j["sensitivity"]["with meeting verdicts|expert votes removed"]["pooled"]
    assert (full["a"]["n"], full["b"]["n"]) == (3, 4)
    assert j["gold_counts"]["not_in_pool"] == {"1": 0, "2": 0}


def test_build_frame_refuses_a_round_1_item_missing_from_the_pool(tmp_path):
    g = _gold()
    root = _split_root(tmp_path, g)
    pool = _pool(g)
    with pytest.raises(ValueError, match="round-1 gold items are not in the pool"):
        ts.build_frame(g, pool[pool.label_uid_schema != "sidewalk_a:1"], root)
    f = ts.build_frame(g, pool[pool.label_uid_schema != "sidewalk_b:8"], root)  # a round-2 decoy keeps its tallies
    assert f.set_index("label_uid").loc["city-b:8", ["h_agree", "h_dis", "n_expert_votes"]].tolist() == [4, 0, 0]


def _composition_frame():
    """Type A: kept 18/20, dropped 9/10 (90% each); type B: kept 5/10, dropped 10/20 (50% each); type C: kept only.
    Pooled, kept (28/35) beats dropped (19/30) only because A is mostly kept and B mostly dropped."""
    rows = []

    def add(t, rnd, kept, ok, n):
        rows.extend({"label_type": t, "round": rnd, "ai_vote": "Disagree", "rule_margin": "incorrect",
                     "rule_symmetric3": "incorrect" if kept else "", "expert": "incorrect" if i < ok else "correct"}
                    for i in range(n))
    add("CurbRamp", "1", True, 9, 10)
    add("CurbRamp", "2", True, 9, 10)
    add("CurbRamp", "2", False, 9, 10)
    add("Obstacle", "1", True, 5, 10)
    add("Obstacle", "1", False, 10, 20)
    add("Crosswalk", "2", True, 5, 5)
    return pd.DataFrame(rows)


def test_mh_diff_removes_a_composition_effect():
    f = _composition_frame()
    ok, kept = (f.expert == "incorrect").to_numpy(), (f.rule_symmetric3 == "incorrect").to_numpy()
    assert ok[kept].mean() - ok[~kept].mean() == pytest.approx(28 / 35 - 19 / 30)
    d, used, n = ts.mh_diff(ok, kept, f.label_type)
    assert d == pytest.approx(0.0) and used == 2 and n == 60  # Crosswalk has no dropped row and no weight
    r = ts.mh_boot(ok, kept, f.label_type, 500, 2)
    assert r["lo"] < 0 < r["hi"] and (r["strata"], r["strata_used"], r["n"], r["n_used"]) == (3, 2, 65, 60)
    assert ts.mh_boot(ok, kept, f.label_type, 500, 2) == r
    # unequal within-type differences: weights n_kept * n_dropped / n
    ok2 = ok.copy()
    ok2[(f.label_type == "Obstacle").to_numpy() & ~kept] = [True] * 5 + [False] * 15  # Obstacle dropped 5/20
    d2, _, _ = ts.mh_diff(ok2, kept, f.label_type)
    wa, wb = 20 * 10 / 30, 10 * 20 / 30
    assert d2 == pytest.approx((wa * 0.0 + wb * (0.5 - 0.25)) / (wa + wb))


def test_p1_stratified_reads_only_p1_rows():
    f = _composition_frame()
    st = ts.p1_stratified(f, n_boot=0)
    assert st["type"]["diff"] == pytest.approx(0.0) and st["type"]["lo"] is None
    # type x round: CurbRamp round 1 has no dropped row, so only CurbRamp/2 and Obstacle/1 carry weight
    assert (st["type_round"]["strata"], st["type_round"]["strata_used"]) == (4, 2)
    extra = pd.DataFrame([{"label_type": "CurbRamp", "round": "2", "ai_vote": ai, "rule_margin": m,
                           "rule_symmetric3": "", "expert": "correct"}
                          for ai, m in (("Agree", "incorrect"), ("Disagree", "correct"), ("Disagree", ""))
                          for _ in range(5)])
    assert ts.p1_stratified(pd.concat([f, extra], ignore_index=True), n_boot=0) == st


def test_strat_diff_boot_keeps_the_point_and_is_seeded():
    ok = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    a = [True, False] * 6
    s = ["x"] * 6 + ["y"] * 6
    d, lo, hi = ts.strat_diff_boot(ok, a, s, 500, 3)
    assert d == pytest.approx(ts.diff_boot(ok, a, 0)[0]) and lo <= d <= hi
    assert ts.strat_diff_boot(ok, a, s, 500, 3) == (d, lo, hi)
    # one stratum all A, the other all B: every resample keeps both groups whole, so the interval is that of the
    # two stratum means, never a skipped draw
    d2, lo2, hi2 = ts.strat_diff_boot([1] * 6 + [0] * 6, [True] * 6 + [False] * 6, s, 200, 1)
    assert (d2, lo2, hi2) == (1.0, 1.0, 1.0)


def test_pool_expert_counts_votes_and_placements():
    p = pd.DataFrame({"ai_vote": ["Disagree"] * 5 + ["Agree"], "h_agree": ["0", "0", "1", "0", "2", "0"],
                      "h_dis": ["2", "3", "0", "1", "2", "5"], "expert1": ["", "", "", "", "", "Disagree"],
                      "expert2": ["", "Disagree", "", "", "", ""], "expert1_labeled": ["f", "f", "t", "f", "f", "f"]})
    e = ts.pool_expert(p)
    assert (e["n"], e["touched"], e["margin_incorrect"], e["touched_margin_incorrect"]) == (5, 2, 2, 1)
    # removing the Disagree of row 2 leaves (0, 2): still a margin target; row 1 keeps (0, 2)
    assert e["removed"]["rules"]["margin"] == 2 and e["removed"]["rules"]["symmetric3"] == 0
    assert e["removed"]["classes"]["1 vote"] == 2 and e["removed"]["classes"]["tie"] == 1


def test_weighted_p1_ignores_rows_outside_its_strata():
    w = {"1": {(grp, t): 100 for grp in ("voted_disagree", "lone_disagree") for t in ts.LABEL_TYPES},
         "2": {(c, s, "NoCurbRamp"): 100 for c in ("city-a", "city-b") for s in ("margin_incorrect", "light_voted")}}
    base = ts.weighted_p1(_p1_frame(), w)
    rows = []
    for rnd, grp, stratum, ai in (("1", "lone_disagree", "lone", "Disagree"), ("2", "", "light_voted", "Disagree"),
                                  ("1", "voted_disagree", "", "Agree"), ("2", "", "margin_incorrect", "Agree")):
        t = "CurbRamp" if rnd == "1" else "NoCurbRamp"
        rows += [{"round": rnd, "ai_vote": ai, "city_id": "city-a", "label_type": t, "group_r1": grp,
                  "stratum": stratum, "rule_margin": "incorrect", "rule_symmetric3": "", "expert": "correct"}] * 6
    out = ts.weighted_p1(pd.concat([_p1_frame(), pd.DataFrame(rows)], ignore_index=True), w)
    assert ts.dumps(out) == ts.dumps(base)


def test_slice_rows_reads_only_ai_disagrees_and_p1_rows():
    g = pd.DataFrame({"label_uid": [f"city-a:{i}" for i in range(7)], "ai_vote": "Disagree",
                      "label_type": "CurbRamp", "severity": ["1"] * 4 + ["2"] * 2 + ["3"],
                      "tags": ["narrow"] * 4 + [""] * 3, "expert": ["incorrect"] * 3 + ["correct"] * 4,
                      "rule_margin": "incorrect", "rule_symmetric3": ["incorrect"] * 3 + [""] * 4})
    train = pd.DataFrame({"label_type": ["CurbRamp"] * 3, "severity": ["1", "1", "2"], "tags": ["[]"] * 3})
    base = ts.slice_rows(g, train, min_n=3)
    extra = g.iloc[:4].assign(label_uid=[f"city-a:{50 + i}" for i in range(4)], expert="correct")
    agree = extra.assign(ai_vote="Agree")
    margin_correct = extra.assign(rule_margin="correct", label_uid=[f"city-a:{60 + i}" for i in range(4)])
    out = ts.slice_rows(pd.concat([g, agree], ignore_index=True), train, min_n=3)
    assert ts.dumps(out) == ts.dumps(base)
    out2 = ts.slice_rows(pd.concat([g, margin_correct], ignore_index=True), train, min_n=3)
    assert ts.dumps(out2["p1_severity"]) == ts.dumps(base["p1_severity"])
    assert ts.dumps(out2["p1_type"]) == ts.dumps(base["p1_type"])
    t = {r["slice"]: r for r in base["p1_type"]}["CurbRamp"]
    assert (t["kept"]["k"], t["kept"]["n"], t["dropped"]["k"], t["dropped"]["n"]) == (3, 3, 0, 4)


def test_git_commit_marks_a_dirty_tree_and_survives_no_git(monkeypatch):
    class R:
        def __init__(self, out, code=0):
            self.stdout, self.returncode = out, code

    monkeypatch.setattr(ts.subprocess, "run", lambda cmd, **k: R("abc\n") if "rev-parse" in cmd else R(" M x.py\n"))
    assert ts._git_commit() == "abc-dirty"
    monkeypatch.setattr(ts.subprocess, "run", lambda cmd, **k: R("abc\n") if "rev-parse" in cmd else R(""))
    assert ts._git_commit() == "abc"

    def no_git(cmd, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(ts.subprocess, "run", no_git)
    assert ts._git_commit() == "unknown"


def test_refuse_private_out_holds_without_git(tmp_path, monkeypatch):
    def no_git(path, what):
        raise FileNotFoundError("git")
    monkeypatch.setattr(ts.adj, "refuse_in_checkout", no_git)
    with pytest.raises(SystemExit, match="inside the repository"):
        ts.refuse_private_out(os.path.join(ROOT, "x"), "verdicts")
    ts.refuse_private_out(str(tmp_path / "private"), "verdicts")  # outside the repo: allowed


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
    with pytest.raises(SystemExit, match="refusing to write the per-label verdicts inside the repository"):
        ts.main(args + ["--out-dir", os.path.join(ROOT, "tmp-should-not-exist"), "--out", str(tmp_path / "pub")])
    assert not os.path.exists(os.path.join(ROOT, "tmp-should-not-exist"))
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


def test_both_primaries_are_pooled_primary_estimates():
    assert ts.ROLES == {"P1": "primary", "P2": "primary", "P1_unanimous3": "descriptive"}


def test_render_primary_says_where_the_flagged_rows_come_from():
    j, _ = _committed()
    text = ts.render_primary(j)
    assert ("Of the pooled P2's 13 flagged rows, 2 are from round 1 and 11 from round 2; with 2 flagged round-1 rows "
            "the rounds cannot be compared.") in text
    for n1, n2, tail in ((1, 1, "with 1 flagged round-1 row and 1 flagged round-2 row the rounds cannot be compared."),
                         (6, 7, "6 are from round 1 and 7 from round 2.")):
        k = json.loads(json.dumps(j))
        k["primary"]["P2"]["1"]["a"]["n"], k["primary"]["P2"]["2"]["a"]["n"] = n1, n2
        assert tail in ts.render_primary(k)
        assert "mostly" not in ts.render_primary(k)


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


def test_committed_json_thin_cells_carry_no_verdict_count():
    """A thin cell can consist only of items from non-public deployments (round 1's decoys hold a one-item rule
    cell), so it publishes its size and no count of verdicts."""
    j, _ = _committed()
    s = j["slices"]
    mn = s["min_n"]
    for by in ("severity", "tag"):
        for r in s[f"ai_precision_{by}"]:
            assert (r["k"] is None) == (r["n"] < mn), r
    for key in ("p1_severity", "p1_type"):
        for r in s[key]:
            for grp in ("kept", "dropped"):
                assert (r[grp]["k"] is None) == (r[grp]["n"] < mn), r
    cells = []
    for pop in ("ai_disagree", "decoys"):
        for r, rows in j["rules"][pop]["rows"].items():
            cells += [((pop, r, rule, c), rows[rule][c]) for rule in ts.TALLY_RULES for c in ("incorrect", "correct",
                                                                                                "agree")]
    det = [(f"{v}/{c}", rows) for v, by in j["detector"].items() for c, rows in by.items()]
    det += [("P2", j["primary"]["P2"])] + [(k, r) for k, r in j["sensitivity"].items() if r["label"] == "P2"]
    for name, rows in det:
        for r in ts.ROUNDS:
            cells += [((name, r, x), rows[r][x]) for x in ts.DETECTOR_CELLS]
            if rows[r]["a"]["k"] is None or rows[r]["b"]["k"] is None:  # a difference or p would give the k back
                assert rows[r]["diff"] is None and rows[r]["lo"] is None and rows[r]["fisher_p"] is None, (name, r)
    assert len(cells) >= 300  # 108 rule cells, 180 detector cells, P2 and its two sensitivity rows
    for key, cell in cells:
        if cell["n"] < ts.MIN_CELL:  # thin; a larger cell can be blank too (its count would give a thin one)
            assert cell["k"] is None, key
        if cell["k"] is None:
            assert cell["rate"] is None and cell["lo"] is None and cell["hi"] is None, key
    rd = j["rules"]["decoys"]["rows"]
    assert rd["1"]["unanimous3"]["incorrect"]["n"] == 1 and rd["1"]["unanimous3"]["agree"]["k"] is None
    assert j["detector"]["cl-drop"]["correct"]["1"]["recall"]["n"] == 1


def test_doc_prose_quotes_the_type_stratified_p1():
    j, doc = _committed()
    prose = " ".join(re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED: \w+ -->", "", doc, flags=re.S).split())
    st = j["primary"]["P1_stratified"]
    for k in ("type", "type_round"):
        assert f"{100 * st[k]['diff']:+.1f}" in prose
    sp = {r["slice"]: r for r in j["slices"]["p1_type"]}["SurfaceProblem"]["dropped"]
    assert f"{sp['k']}/{sp['n']}" in prose and f"{100 * sp['lo']:.1f}" in prose


VERDICTS = os.path.expanduser("~/Data/sidewalk-validator-ai-target-scoreboard-2026-11/verdicts.csv")


@pytest.mark.skipif(not os.path.exists(VERDICTS), reason="private per-label verdicts not on this machine")
def test_committed_json_recomputes_from_the_private_verdicts():
    """On the machine that ran ``score``: recompute the primaries, the type-stratified P1 and P1 by type from the
    private verdicts and compare with the committed JSON (the other committed-JSON tests only read it). Skipped in CI,
    which has no private data."""
    j, _ = _committed()
    if ts.rl.sha256_file(VERDICTS) != j["provenance"]["verdicts_sha256"]:
        pytest.skip("the private verdicts are not the ones the committed JSON was built from")
    v = pd.read_csv(VERDICTS, dtype=str, keep_default_na=False)
    for c in ("masked", "train_row", "wrong", "cl_flagged") + tuple(f"touch_{x}" for x in ts.VARIANTS):
        v[c] = v[c] == "True"
    for c in ("h_agree", "h_dis"):
        v[c] = v[c].astype(int)
    g = v[v.masked]
    p1, p2 = ts.p1_rows(g, n_boot=0)["pooled"], ts.p2_rows(g, "cl-drop", n_boot=0)["pooled"]
    for mine, theirs in ((p1, j["primary"]["P1"]["pooled"]), (p2, j["primary"]["P2"]["pooled"])):
        for grp in ("a", "b"):
            assert (mine[grp]["k"], mine[grp]["n"]) == (theirs[grp]["k"], theirs[grp]["n"])
        assert mine["diff"] == pytest.approx(theirs["diff"])
    st = ts.p1_stratified(g, n_boot=0)
    for k in ("type", "type_round"):
        assert st[k]["diff"] == pytest.approx(j["primary"]["P1_stratified"][k]["diff"])
    # verdicts.csv carries no severity or tags, so only P1 by type is recomputed (blank columns stand in)
    mine = ts.slice_rows(g.assign(severity="", tags=""), pd.DataFrame(columns=["label_type", "severity", "tags"]))
    for key in ("p1_type",):
        assert [(r["kept"]["n"], r["dropped"]["n"], r["kept"]["k"], r["dropped"]["k"]) for r in mine[key]] == \
            [(r["kept"]["n"], r["dropped"]["n"], r["kept"]["k"], r["dropped"]["k"]) for r in j["slices"][key]]



# ---------------------------------------------------------------------------------------------------------------------
# Thin cells cannot be worked out from published counts (rule and detector blocks)
# ---------------------------------------------------------------------------------------------------------------------
# A published count is a linear function of "atoms" (counts of rows sharing every attribute the tables read). For
# each thin cell (1 to MIN_CELL - 1 items) the check asks whether its count lies in the span of the published
# counts and sizes but not in the span of the sizes alone, after dropping the atoms a published zero fixes. The
# integer-program check (SciPy, so skipped in CI) also uses that counts are whole and non-negative.

DET_CELLS = {"a": ("t", 1, None), "b": ("t", 0, None), "recall": ("w", 1, ("t", 1)),
             "false_alarm": ("w", 0, ("t", 1)), "kept_accuracy": ("t", 0, ("w", 0))}


def _vec(atoms, pred):
    return np.array([1.0 if pred(a) else 0.0 for a in atoms])


def _detector_system(variants, extra=()):
    """Atoms (round, target class, wrong, touched by each variant); equations from ``{variant: {class: rows}}``
    (:func:`ts.detector_rows`) plus ``extra`` rows given as ``(variant, class, rows, label)``."""
    vs = list(variants)
    atoms = [dict(r=r, c=c, w=w, t=dict(zip(vs, t))) for r in ("1", "2") for c in ("incorrect", "correct")
             for w in (0, 1) for t in itertools.product((0, 1), repeat=len(vs))]
    sizes, counts, thin = [], [], []

    def val(a, v, key):
        return a["t"][v] if key == "t" else a["w"]

    def add(v, c, rows, label):
        for r in ts.ROUNDS:
            for x, (key, want, also) in DET_CELLS.items():
                def grp(a, r=r, key=key, want=want):
                    return (c == "all" or a["c"] == c) and (r == "pooled" or a["r"] == r) and val(a, v, key) == want
                if also is None:
                    def hit(a, grp=grp):
                        return grp(a) and a["w"] == 1
                else:
                    def hit(a, grp=grp, also=also):
                        return grp(a) and val(a, v, also[0]) == also[1]
                cell = rows[r][x]
                sizes.append((_vec(atoms, grp), cell["n"]))
                if cell["k"] is not None:
                    counts.append((_vec(atoms, hit), cell["k"]))
                if 0 < cell["n"] < ts.MIN_CELL:
                    thin.append((f"{label} {v}/{c}/{r}/{x} n={cell['n']}", _vec(atoms, hit)))
    for v, by_class in variants.items():
        for c, rows in by_class.items():
            add(v, c, rows, "detector")
    for v, c, rows, label in extra:
        add(v, c, rows, label)
    return atoms, sizes, counts, thin


def _rules_system(rows, p1=(), hmax=12):
    """Atoms (round, the verdicts every rule gives together, as :func:`ts.tally_verdicts` can give them, and the
    expert's verdict); equations from :func:`ts.rule_rows`'s ``rows`` and from ``p1``, ``(rule, P1 rows)`` pairs
    that read the same rows (the AI Disagrees)."""
    tv = ts.tally_verdicts(*zip(*[(a, d) for a in range(hmax) for d in range(hmax)]))
    rules = list(ts.TALLY_RULES)
    atoms = [dict(r=r, v=dict(zip(rules, v)), e=e) for r in ("1", "2")
             for v in sorted(set(map(tuple, tv.values.tolist()))) for e in ("incorrect", "correct")]
    sizes, counts, thin = [], [], []

    def ing(a, r):
        return r == "pooled" or a["r"] == r
    for r in ts.ROUNDS:
        x = rows[r]
        sizes.append((_vec(atoms, lambda a: ing(a, r)), x["n"]))
        if x["expert_incorrect"]["k"] is not None:
            counts.append((_vec(atoms, lambda a: ing(a, r) and a["e"] == "incorrect"), x["expert_incorrect"]["k"]))
        for rule in rules:
            for c in ("incorrect", "correct", "agree"):
                if c == "agree":
                    def grp(a, rule=rule):
                        return ing(a, r) and a["v"][rule] != ""

                    def hit(a, grp=grp, rule=rule):
                        return grp(a) and a["v"][rule] == a["e"]
                else:
                    def grp(a, rule=rule, c=c):
                        return ing(a, r) and a["v"][rule] == c

                    def hit(a, grp=grp, c=c):
                        return grp(a) and a["e"] == c
                cell = x[rule][c]
                sizes.append((_vec(atoms, grp), cell["n"]))
                if cell["k"] is not None:
                    counts.append((_vec(atoms, hit), cell["k"]))
                if 0 < cell["n"] < ts.MIN_CELL:
                    thin.append((f"rules {rule}/{c}/{r} n={cell['n']}", _vec(atoms, hit)))
    for rule, prow in p1:
        for r in ts.ROUNDS:
            for name, kept in (("a", True), ("b", False)):
                def grp(a, r=r, kept=kept, rule=rule):
                    return ing(a, r) and a["v"]["margin"] == "incorrect" and (a["v"][rule] == "incorrect") == kept
                cell = prow[r][name]
                sizes.append((_vec(atoms, grp), cell["n"]))
                if cell["k"] is not None:
                    counts.append((_vec(atoms, lambda a, grp=grp: grp(a) and a["e"] == "incorrect"), cell["k"]))
    return atoms, sizes, counts, thin


def _committed_systems(j):
    """The rule and detector systems of a scoreboard.json, with every other block that publishes the same counts:
    P2 and its masked-only sensitivity row, check B, the out-of-fold ranking rows' sizes, P1 and its twin."""
    det = _detector_system(j["detector"], [("cl-drop", "all", j["primary"]["P2"], "P2"),
                                           ("cl-drop", "all", j["sensitivity"]["masked-only|n/a (train targets)"],
                                            "sensitivity")])
    atoms, sizes, counts, _ = det
    for v, b in j["check_b"].items():
        sizes += [(_vec(atoms, lambda a: True), b["n_rated_train"]),
                  (_vec(atoms, lambda a: a["w"] == 0), b["target_matches_expert"]),
                  (_vec(atoms, lambda a, v=v: a["t"][v] == 1), b["touched"]),
                  (_vec(atoms, lambda a, v=v: a["t"][v] == 0), b["untouched"])]
        counts += [(_vec(atoms, lambda a, v=v: a["t"][v] == 1 and a["w"] == 1), b["touched_overturned"]),
                   (_vec(atoms, lambda a, v=v: a["t"][v] == 0 and a["w"] == 1), b["untouched_overturned"])]
    for row in j["ranking"]:
        if row["score"].startswith("oof"):
            sizes += [(_vec(atoms, lambda a, c=row["target_class"]: a["c"] == c), row["n"]),
                      (_vec(atoms, lambda a, c=row["target_class"]: a["c"] == c and a["w"] == 1), row["n_wrong"])]
    p1 = (("symmetric3", j["primary"]["P1"]), ("unanimous3", j["primary"]["P1_unanimous3"]))
    return {"detector": det, "decoys": _rules_system(j["rules"]["decoys"]["rows"]),
            "ai_disagree": _rules_system(j["rules"]["ai_disagree"]["rows"], p1)}


def _determined(eqs, target):
    """Whether the equations ``[(vector over atoms, value)]`` fix ``target``'s value (linear algebra over the
    reals, after dropping the atoms of every published zero)."""
    a = np.array([c for c, _ in eqs])
    zero = np.zeros(a.shape[1], bool)
    for c, v in eqs:
        if v == 0:
            zero |= c > 0
    t = target[~zero]
    if not t.any():
        return True
    a = a[:, ~zero]
    return np.linalg.matrix_rank(np.vstack([a, t])) == np.linalg.matrix_rank(a)


def _linear_leaks(system):
    _, sizes, counts, thin = system
    return [label for label, t in thin if _determined(sizes + counts, t) and not _determined(sizes, t)]


def test_determined_sees_a_difference_and_a_sum():
    atoms = [0, 1, 2]
    sizes = [(np.array([1.0, 1, 0]), 3)]
    t = np.array([1.0, 0, 0])
    assert not _determined(sizes, t)
    assert _determined(sizes + [(np.array([0.0, 1, 0]), 2)], t)  # 3 - 2
    assert _determined(sizes + [(np.array([0.0, 1, 0]), 0)], t)  # a published zero
    assert not _determined(sizes + [(np.array([0.0, 1, 1]), 2)], t)
    assert len(atoms) == 3


def test_committed_json_no_thin_count_can_be_worked_out():
    """Every thin cell of the rule and detector blocks keeps its count hidden: no combination of published counts
    (with the sizes, P2, check B and P1, which publish counts over the same rows) gives it."""
    j, _ = _committed()
    systems = _committed_systems(j)
    assert sum(len(s[3]) for s in systems.values()) >= 30  # the round-1 decoy and correct-target cells are in
    for name, system in systems.items():
        assert _linear_leaks(system) == [], name


def test_committed_json_no_thin_count_is_pinned_by_whole_numbers():
    """The same as an integer program: counts are whole and non-negative, so a pooled count can pin a round's
    when the bounds are tight. The thin cell's range given the sizes alone must stay a range."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    j, _ = _committed()

    def span(atoms, eqs, t):
        a = np.array([c for c, _ in eqs])
        b = np.array([v for _, v in eqs], float)
        out = []
        for sign in (1, -1):
            r = scipy_opt.milp(sign * t, constraints=scipy_opt.LinearConstraint(a, b, b),
                               integrality=np.ones(len(atoms)), bounds=scipy_opt.Bounds(0, np.inf))
            assert r.success, r.message
            out.append(round(sign * r.fun))
        return out
    pinned = []
    for name, (atoms, sizes, counts, thin) in _committed_systems(j).items():
        for label, t in thin:
            lo, hi = span(atoms, sizes, t)
            lo2, hi2 = span(atoms, sizes + counts, t)
            if lo < hi and lo2 == hi2:
                pinned.append(label)
    assert pinned == []


def _decoy_frame(r1_abstain=True, r2_incorrect=6):
    """Decoys only, from vote tallies: round 1 has 10 (3, 0) items, 2 (0, 3) items (a thin `incorrect` cell for
    every rule) and, unless ``r1_abstain`` is False, 3 (0, 0) items that no rule gives a target; round 2 has 10
    (3, 0) items, ``r2_incorrect`` (0, 3) items and 2 (1, 1) items (no target)."""
    rows = ([("1", 3, 0, "correct")] * 9 + [("1", 3, 0, "incorrect")] + [("1", 0, 3, "incorrect"),
                                                                           ("1", 0, 3, "correct")]
            + ([("1", 0, 0, "incorrect")] * 3 if r1_abstain else [])
            + [("2", 3, 0, "correct")] * 8 + [("2", 3, 0, "incorrect")] * 2
            + [("2", 0, 3, "incorrect")] * (r2_incorrect - 1) + [("2", 0, 3, "correct")] + [("2", 1, 1, "correct")] * 2)
    g = pd.DataFrame(rows, columns=["round", "h_agree", "h_dis", "expert"])
    g["ai_vote"] = "Agree"
    tv = ts.tally_verdicts(g.h_agree, g.h_dis)
    for r in ts.TALLY_RULES:
        g[f"rule_{r}"] = tv[r].to_numpy()
    return g


def test_rule_rows_blank_agree_and_the_pooled_cell_with_a_thin_cell():
    rows = ts.rule_rows(_decoy_frame(), "decoys")["rows"]
    for rule in ts.TALLY_RULES:
        k = {(c, r): rows[r][rule][c]["k"] for c in ("incorrect", "correct", "agree") for r in ts.ROUNDS}
        assert rows["1"][rule]["incorrect"]["n"] == 2 and k["incorrect", "1"] is None
        assert k["agree", "1"] is None  # agree - correct would give the thin cell
        assert k["incorrect", "pooled"] is None and k["agree", "pooled"] is None  # pooled - round 2 would
        assert k["correct", "1"] == 9 and k["incorrect", "2"] == 5 and k["agree", "2"] == 13  # abstentions in round 1
        assert rows["1"][rule]["incorrect"]["rate"] is None and rows["pooled"][rule]["agree"]["lo"] is None
    assert _linear_leaks(_rules_system(rows)) == []


def test_rule_rows_blank_the_other_class_when_no_item_is_left_without_a_target():
    """With no abstention in a round, the expert-incorrect count is the `incorrect` cell's k plus the `correct`
    cell's n - k, so the `correct` cell would give the thin one away."""
    rows = ts.rule_rows(_decoy_frame(r1_abstain=False), "decoys")["rows"]
    assert rows["1"]["n"] == 12 and rows["1"]["expert_incorrect"]["k"] == 2
    for rule in ts.TALLY_RULES:
        assert rows["1"][rule]["correct"]["k"] is None and rows["1"][rule]["correct"]["n"] == 10
        assert rows["pooled"][rule]["correct"]["k"] is None
        assert rows["2"][rule]["correct"]["k"] == 8
    assert _linear_leaks(_rules_system(rows)) == []


def test_rule_rows_hide_the_pooled_cell_when_both_rounds_are_thin():
    rows = ts.rule_rows(_decoy_frame(r2_incorrect=3), "decoys")["rows"]
    for rule in ts.TALLY_RULES:
        assert rows["pooled"][rule]["incorrect"]["n"] == 5 and rows["pooled"][rule]["incorrect"]["k"] is None
        assert rows["pooled"][rule]["correct"]["k"] == 17


def _detector_frame():
    """Train rows by round, target class, wrong and touched. `cl-drop` (and `cl-flip`, as in the real run) touches
    both classes, and the `correct` class has 1 + 2 wrong targets, so its rows are thin; `symmetric` and
    `unanimous3` never touch a `correct` target, so its rows' counts follow from their sizes."""
    spec = [  # (round, target class, touched by cl-*, touched by the rules, wrong, rows)
        ("1", "incorrect", 1, 1, 1, 3), ("1", "incorrect", 1, 1, 0, 2), ("1", "incorrect", 0, 0, 1, 2),
        ("1", "incorrect", 0, 0, 0, 3),
        ("2", "incorrect", 1, 1, 1, 3), ("2", "incorrect", 1, 1, 0, 3), ("2", "incorrect", 0, 0, 1, 2),
        ("2", "incorrect", 0, 0, 0, 4),
        ("1", "correct", 1, 0, 1, 1), ("1", "correct", 1, 0, 0, 4), ("1", "correct", 0, 0, 0, 5),
        ("2", "correct", 1, 0, 1, 1), ("2", "correct", 1, 0, 0, 4), ("2", "correct", 0, 0, 1, 1),
        ("2", "correct", 0, 0, 0, 5)]
    rows = [dict(round=r, target=c, wrong=bool(w), **{"touch_cl-drop": bool(t), "touch_cl-flip": bool(t),
                                                      "touch_symmetric": bool(u), "touch_unanimous3": bool(u)})
            for r, c, t, u, w, n in spec for _ in range(n)]
    g = pd.DataFrame(rows)
    g["train_row"] = True
    return g


def test_detector_rows_hide_the_class_rows_and_keep_the_pooled_row():
    g = _detector_frame()
    det = {v: ts.detector_rows(g, v, n_boot=0) for v in ts.VARIANTS}
    cl_ = det["cl-drop"]
    assert cl_["correct"]["1"]["recall"]["n"] == 1 and cl_["correct"]["2"]["recall"]["n"] == 2
    for c in ("incorrect", "correct"):  # all - correct would give incorrect, so both go
        for r in ts.ROUNDS:
            row = cl_[c][r]
            assert all(row[x]["k"] is None for x in ts.DETECTOR_CELLS), (c, r)
            assert row["diff"] is None and row["fisher_p"] is None
        assert cl_[c]["heterogeneity"]["did"] is None
    assert all(cl_["all"][r]["a"]["k"] is not None for r in ts.ROUNDS)  # P2 and check B publish the pooled row
    raw = ts.p2_rows(g, "cl-drop", "correct", n_boot=0)
    assert raw["heterogeneity"]["did"] is not None  # sealing, not did_boot, blanks it
    sy = det["symmetric"]
    assert sy["incorrect"]["pooled"]["a"]["k"] is not None and sy["all"]["pooled"]["b"]["k"] is not None
    c1 = sy["correct"]["1"]
    assert c1["a"]["n"] == 0 and c1["b"]["k"] == 1 and c1["recall"]["k"] is None and c1["fisher_p"] is None
    assert _linear_leaks(_detector_system(det)) == []
    j = {"detector": det, "gold_counts": {"train_rows_by_round": {"1": 1}}}
    text = ts.render_detector(j)
    assert "| `cl-drop` | correct | 21 | n 10 | n 11 | n/a | n/a |" in text


def test_seal_detector_without_the_class_rows_leaks():
    """The check the sealing is for: blanking only the thin cells leaves them computable."""
    g = _detector_frame()
    raw = {v: {c or "all": ts.p2_rows(g, v, c, n_boot=0) for c in ("incorrect", "correct", None)}
           for v in ts.VARIANTS}
    for v in raw.values():
        for rows in v.values():
            for r in ts.ROUNDS:
                for x in ts.DETECTOR_CELLS:
                    if rows[r][x]["n"] < ts.MIN_CELL:
                        rows[r][x] = ts._size_only(rows[r][x])
    assert _linear_leaks(_detector_system(raw)) != []


def test_provenance_row_names_a_saved_city_list():
    live = ts.render_provenance({"provenance": _prov()})
    assert "| public-deployment list | live `/v3/api/cities` response, read t (2 public) |" in live
    saved = ts.render_provenance({"provenance": {**_prov(), "cities_response": "saved"}})
    assert "| public-deployment list | saved `/v3/api/cities` response, read t (2 public) |" in saved
    assert "live" not in saved.split("public-deployment list")[1]


def test_refuse_private_out_follows_a_symlink_into_the_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(ts.adj, "refuse_in_checkout", lambda path, what: None)
    link = tmp_path / "looks-private"
    link.symlink_to(ROOT)
    with pytest.raises(SystemExit, match="inside the repository"):
        ts.refuse_private_out(str(link / "out"), "verdicts")


def test_rule_rows_show_an_empty_cell_by_size():
    g = _decoy_frame()
    g = g[~((g["round"] == "2") & (g.h_dis == 3))]
    rows = ts.rule_rows(g, "decoys")["rows"]
    for rule in ts.TALLY_RULES:
        assert rows["2"][rule]["incorrect"] == {"n": 0, "k": None, "rate": None, "lo": None, "hi": None}


def _one_variant_frame(spec):
    rows = [dict(round=r, target=c, wrong=bool(w), **{f"touch_{v}": bool(t) for v in ts.VARIANTS})
            for r, c, t, w, n in spec for _ in range(n)]
    return pd.DataFrame(rows).assign(train_row=True)


def test_detector_rows_close_over_rounds():
    """Only round 1's `correct` row is thin; round 2's and the pooled one are not, so one of them goes too, and the
    `incorrect` rows follow (the row over both classes stays)."""
    inc = [(r, "incorrect", t, w, 4) for r in ("1", "2") for t in (0, 1) for w in (0, 1)]
    g = _one_variant_frame(inc + [("1", "correct", 1, 1, 1), ("1", "correct", 1, 0, 1), ("1", "correct", 0, 1, 1),
                                  ("1", "correct", 0, 0, 5), ("2", "correct", 1, 1, 3), ("2", "correct", 1, 0, 5),
                                  ("2", "correct", 0, 1, 2), ("2", "correct", 0, 0, 6)])
    det = ts.detector_rows(g, "cl-drop", n_boot=0)
    assert det["correct"]["1"]["a"]["n"] == 2 and det["correct"]["pooled"]["recall"]["n"] == 7
    assert det["correct"]["1"]["a"]["k"] is None
    assert det["correct"]["2"]["a"]["k"] is None or det["correct"]["pooled"]["a"]["k"] is None
    assert all(det["all"][r]["a"]["k"] is not None for r in ts.ROUNDS)
    assert _linear_leaks(_detector_system({"cl-drop": det})) == []


def test_detector_rows_never_hide_the_pooled_row_over_both_classes():
    """Check B publishes each variant's pooled row over both classes outside this table, so it is never hidden
    here. (When the other class's counts follow from its sizes, the thin class's pooled count then follows from
    check B; the committed-JSON test above checks that this does not happen in the real run.)"""
    g = _one_variant_frame([(r, "incorrect", 0, w, 6) for r in ("1", "2") for w in (0, 1)]
                           + [(r, "correct", t, w, n) for r in ("1", "2")
                              for t, w, n in ((1, 1, 1), (1, 0, 5), (0, 1, 1), (0, 0, 5))])
    det = ts.detector_rows(g, "cl-drop", n_boot=0)
    assert det["incorrect"]["pooled"]["a"]["n"] == 0  # no incorrect target touched: its counts follow from sizes
    assert det["all"]["pooled"]["a"]["k"] == 2 and det["all"]["1"]["a"]["k"] is None
