"""analysis/backbone_comparison.py (issue #119): the data view, the run-consistency check, the paired within-city
difference, the pre-registered rule, and the report on a tiny synthetic run directory."""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from analysis import backbone_comparison as bc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARES = os.path.join(REPO, "docs", "deployment-prior-2026-11", "shares.json")
PRIVATE = "hiddenvale-xx"   # a made-up deployment the public list leaves out
PUBLIC = ["amsterdam", "chicago-il", "seattle-wa", "pubtown-wa", "teaneck-nj"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key, seed, lr, rand", [("dinov2-large", 42, 1e-5, False), ("convnextv2-base", 44, 1e-5, False),
                                                 ("rampnet-convnextv2-base", 42, 3e-5, False),
                                                 ("dinov3-vitl16", 42, 1e-5, True)])
def test_tags_round_trip(key, seed, lr, rand):
    tag = bc.make_tag("bb-", key, seed, lr, rand)
    assert tag.replace("-", "").isalnum()
    assert bc.parse_tag(tag, "bb-") == {"key": key, "seed": seed, "lr": pytest.approx(lr), "random": rand}
    assert bc.parse_tag(tag, "smoke-") is None


# ---------------------------------------------------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------------------------------------------------

def per_type(d=0.0, n_pos=3, wc=0.0, test=0.0, prec=0.95, cov=0.5, n_seeds=3, params=304e6, base_prec=0.95,
             cov_ci=None, prec_ci=None):
    """A ``summarise`` entry. Condition 3's intervals default to the point difference (no width); ``cov_ci`` /
    ``prec_ci`` give ``(lo, hi)`` explicitly, ``(None, None)`` for an unestimable one."""
    dc, dp = cov - 0.5, prec - base_prec
    cl, ch = cov_ci or (dc, dc)
    pl, ph = prec_ci or (dp, dp)
    return {"n_seeds": n_seeds, "seeds": list(bc.SEEDS[:n_seeds]) + list(range(45, 42 + n_seeds)),
            "d_heldout_mean": d, "n_pos_heldout": n_pos, "wc_d_heldout_mean": wc,
            "d_test_mean": test, "agree_prec_mean": prec, "agree_cov_mean": cov, "base_agree_prec_mean": base_prec,
            "base_agree_cov_mean": 0.5, "params": params, "base_params": 304e6,
            "agree_d_cov": dc, "agree_d_cov_lo": cl, "agree_d_cov_hi": ch, "agree_d_prec": dp, "agree_d_prec_lo": pl,
            "agree_d_prec_hi": ph}


def summary(arm, cr, sp):
    return {arm: {"CurbRamp": cr, "SurfaceProblem": sp}}


def status(arm, cr, sp, submitted=()):
    return bc.decide(summary(arm, cr, sp), submitted)["arms"][arm]["status"]


def rows_of(arm, seeds=bc.SEEDS, types=bc.PILOT_TYPES):
    return {(t, arm, s) for t in types for s in seeds}


def test_decide_branches():
    reg = "dinov2-reg-large"
    assert status(reg, per_type(0.012, wc=0.004), per_type(0.0)) == "winner"
    assert status(reg, per_type(0.012, wc=-0.001), per_type(0.0)) == "between-deployment only"
    assert status(reg, per_type(0.012, n_pos=2, wc=0.004), per_type(0.0)) == "null"   # Δ > 0 in 2 of 3 seeds only
    assert status(reg, per_type(0.012, wc=0.004), per_type(-0.006)) == "worse"        # condition 2 fails elsewhere
    assert status(reg, per_type(0.012, wc=0.004, prec=0.93), per_type(0.0)) == "fails the gate"  # precision -0.02
    assert status(reg, per_type(0.012, wc=0.004, cov=0.47), per_type(0.0)) == "fails the gate"   # coverage -3 points
    assert status(reg, per_type(0.004), per_type(0.001)) == "null"
    small = per_type(0.0, params=88e6)
    assert status("convnextv2-base", small, per_type(0.0, params=88e6)) == "non-inferior small"
    # the control cannot win at the single threshold; it needs the doubled one, and is only promoted without an
    # eligible winner
    assert status("convnextv2-base", per_type(0.012, wc=0.004), per_type(0.0)) != "meets the doubled rule"
    s = {**summary("convnextv2-base", per_type(0.021, wc=0.004), per_type(0.0)),
         **summary(reg, per_type(0.004), per_type(0.0))}
    d = bc.decide(s)
    assert d["arms"]["convnextv2-base"]["status"] == "meets the doubled rule" and d["promoted"] == ["convnextv2-base"]
    assert d["recommendation"].startswith("convnextv2-base")
    s[reg] = summary(reg, per_type(0.011, wc=0.003), per_type(0.0))[reg]
    assert bc.decide(s)["recommendation"].startswith(reg)


def test_decide_incomplete_seeds_are_submitted_incomplete_or_not_yet_run():
    reg = "dinov2-reg-large"
    assert status(reg, per_type(0.02, wc=0.01, n_seeds=2), per_type(0.0), rows_of(reg)) == "submitted, incomplete"
    assert status(reg, per_type(0.02, wc=0.01, n_seeds=2), per_type(0.0)) == "not yet run"
    # only the missing seed's row counts: seeds 42 and 43 submitted (and finished), 44 never submitted
    assert status(reg, per_type(0.02, wc=0.01, n_seeds=2), per_type(0.0), rows_of(reg, (42, 43))) == "not yet run"
    d = bc.decide({})
    assert d["recommendation"] == "not yet run"
    assert {a: r["status"] for a, r in d["arms"].items()} == {a: "not yet run" for a in bc.bb.ELIGIBLE}


def test_decide_holds_while_a_submitted_eligible_arm_is_incomplete():
    """PR #121 review S3: dinov3-vitl16 is submitted. While it is incomplete no winner is named (it could win or change
    the tie-break); an eligible arm never submitted never holds the decision but is named as not run."""
    reg, v3 = "dinov2-reg-large", "dinov3-vitl16"
    win = summary(reg, per_type(0.012, wc=0.004), per_type(0.0))
    d = bc.decide(win, rows_of(reg) | rows_of(v3))
    assert d["recommendation"].startswith(f"held: {v3} submitted but incomplete") and d["choice"] is None
    assert f"meets the rule so far: {reg}" in d["recommendation"]
    assert d["arms"][v3]["status"] == "submitted, incomplete" and d["incomplete"] == [v3]
    assert "rampnet-convnextv2-base" in d["not_run"] and "not run: rampnet-convnextv2-base" in d["recommendation"]
    # finished and null: the winner stands
    done = {**win, **summary(v3, per_type(0.002), per_type(0.0))}
    d = bc.decide(done, rows_of(reg) | rows_of(v3))
    assert d["choice"] == reg and d["recommendation"].startswith(reg)
    # no eligible arm met the rule, nothing in flight: keep the baseline, naming what was not run (no "running" forever)
    d = bc.decide(summary(reg, per_type(0.0), per_type(0.0)), rows_of(reg))
    assert d["recommendation"] == ("keep dinov2-large among the arms run (no arm met the rule); not run: "
                                   "rampnet-convnextv2-base, dinov3-vitl16")
    # a submitted arm that failed and will not be rerun stops holding it once dropped
    d = bc.decide(summary(reg, per_type(0.0), per_type(0.0)), rows_of(reg) | rows_of(v3), dropped=[v3])
    assert d["arms"][v3]["status"] == "dropped" and d["recommendation"].startswith("keep dinov2-large")
    assert v3 in d["not_run"]


def test_an_incomplete_explanatory_arm_holds_only_without_an_eligible_winner():
    reg, small = "dinov2-reg-large", "dinov3-vitb16"
    win = summary(reg, per_type(0.012, wc=0.004), per_type(0.0))
    assert bc.decide(win, rows_of(reg) | rows_of(small))["choice"] == reg
    d = bc.decide(summary(reg, per_type(0.0), per_type(0.0)), rows_of(reg) | rows_of(small))
    assert d["recommendation"].startswith(f"held: no eligible arm met the rule, and {small}") and d["choice"] is None


def test_rule_edges_that_survived_mutation():
    """PR #121 review S11: each condition-2 term on its own, the doubled bar exactly, n_pos against n_seeds, and the
    gate's absolute precision floor."""
    reg, ctl = "dinov2-reg-large", "convnextv2-base"
    assert status(reg, per_type(0.012, wc=0.004), per_type(0.0, wc=-0.006)) == "worse"     # within-city held-out
    assert status(reg, per_type(0.012, wc=0.004), per_type(0.0, test=-0.006)) == "worse"   # test
    assert status(reg, per_type(0.012, wc=0.004), per_type(-0.006)) == "worse"             # held-out
    assert status(ctl, per_type(0.016, wc=0.004), per_type(0.0)) == "null"   # 1.6x the bar: not the doubled rule
    assert status(ctl, per_type(0.020, wc=0.004), per_type(0.0)) == "meets the doubled rule"
    four = per_type(0.012, n_pos=4, wc=0.004, n_seeds=4)   # a fourth seed, all positive
    assert status(reg, four, per_type(0.0, n_seeds=4)) == "winner"
    assert not bc._gate_ok(per_type(prec=0.915, base_prec=0.92))   # within 0.01 of the baseline, under 0.92
    assert bc._gate_ok(per_type(prec=0.921, base_prec=0.92))


# ---------------------------------------------------------------------------------------------------------------------
# condition 3 (J17): intervals that include seed variance
# ---------------------------------------------------------------------------------------------------------------------

PI = {"correct": 0.6, "incorrect": 0.4}


def gate_data(n=2400, seed=0):
    """``n`` rows in panoramas of four; Agree on every row (right when the true class is correct)."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    groups = np.array([f"p{i // 4}" for i in range(n)])
    return y, groups, rng


def runs_at(y, rng, probs, shift=None):
    """``{seed: (agree, passed, right)}``: each seed passes the rows whose uniform draw is under its probability;
    ``shift`` gives the arm's per-seed probabilities on the baseline's draws (a paired run on the same rows)."""
    base, arm = {}, {}
    for s, p in probs.items():
        u = rng.random(len(y))
        base[s] = (np.ones(len(y), bool), u < p, ~y)
        if shift is not None:
            arm[s] = (np.ones(len(y), bool), u < shift[s], ~y)
    return base, arm


def test_seed_noise_wider_than_the_margin_passes_condition_3():
    """Per-seed Δ coverage −0.10, +0.05, −0.02: the mean (−0.023) is past −0.02, but the interval, which includes
    seed variance, spans it, so the coverage clause passes."""
    y, g, rng = gate_data()
    base, arm = runs_at(y, rng, {42: 0.7, 43: 0.8, 44: 0.6}, {42: 0.6, 43: 0.85, 44: 0.58})
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=400, seed=1)
    assert r["seeds"] == [42, 43, 44] and r["d_cov"] < -0.02
    assert r["d_cov_lo"] < -0.02 < r["d_cov_hi"] and r["cov_dropped"] == 0
    per = {**per_type(0.012, wc=0.004), **{f"agree_{k}": r[k] for k in bc.GATE_CI_KEYS}, "agree_prec_mean": 0.95}
    assert bc.gate_condition(per) == ("pass", {"precision_floor": "pass", "coverage": "pass", "precision_drop": "pass"})
    assert status("dinov2-reg-large", per, per_type(0.0)) == "winner"


def test_a_consistent_coverage_drop_fails_condition_3():
    y, g, rng = gate_data()
    probs = {42: 0.7, 43: 0.8, 44: 0.6}
    base, arm = runs_at(y, rng, probs, {s: p - 0.06 for s, p in probs.items()})
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=400, seed=1)
    assert r["d_cov_hi"] < -0.02 and r["d_cov"] == pytest.approx(-0.06, abs=0.015)
    assert r["d_prec_lo"] <= 0 <= r["d_prec_hi"] or abs(r["d_prec"]) < 0.01   # precision unchanged by design
    per = {**per_type(0.012, wc=0.004), **{f"agree_{k}": r[k] for k in bc.GATE_CI_KEYS}, "agree_prec_mean": 0.95}
    res, cl = bc.gate_condition(per)
    assert res == "fail" and cl["coverage"] == "fail"
    assert status("dinov2-reg-large", per, per_type(0.0)) == "fails the gate"
    # the per-seed values are the plain gate statistics, and the point estimate is their paired mean
    w = np.where(y, PI["incorrect"] / y.mean(), PI["correct"] / (1 - y.mean()))
    cov = {s: w[arm[s][1]].sum() / w.sum() - w[base[s][1]].sum() / w.sum() for s in probs}
    assert r["d_cov"] == pytest.approx(np.mean(list(cov.values())))
    assert [p["seed"] for p in r["per_seed"]] == [42, 43, 44]
    assert r["per_seed"][0]["cov"] == pytest.approx(w[arm[42][1]].sum() / w.sum())


def test_condition_3_precision_clause_and_floor():
    assert bc.gate_condition(per_type(prec=0.95, prec_ci=(-0.03, -0.011)))[1]["precision_drop"] == "fail"
    assert bc.gate_condition(per_type(prec=0.95, prec_ci=(-0.03, -0.009)))[0] == "pass"
    assert bc.gate_condition(per_type(prec=0.919, base_prec=0.919))[1]["precision_floor"] == "fail"
    # a mean coverage drop past 0.02 with an interval that spans the margin passes; one wholly beyond it fails
    assert bc.gate_condition(per_type(cov=0.47, cov_ci=(-0.08, 0.01)))[0] == "pass"
    assert bc.gate_condition(per_type(cov=0.47, cov_ci=(-0.04, -0.021)))[0] == "fail"


def test_nan_replicates_are_dropped_counted_and_can_leave_the_interval_unestimable():
    y, g, rng = gate_data(n=1200)
    probs = {42: 0.7, 43: 0.8, 44: 0.6}
    base, arm = runs_at(y, rng, probs, probs)
    # the arm gates no Agree at all on seed 44: its precision is undefined there, and most replicates draw seed 44
    a, _, right = arm[44]
    arm[44] = (a, np.zeros(len(y), bool), right)
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    assert r["d_prec"] is None and r["per_seed"][2]["prec"] is None
    assert r["prec_dropped"] > 150 and r["d_prec_lo"] is None and r["d_prec_hi"] is None
    # coverage 0 on that seed is still a coverage: nothing dropped. With three seeds, about 30% of replicates never
    # draw seed 44, so the upper end sits at the other seeds' difference (0 here): one bad seed alone cannot fail it
    assert r["cov_dropped"] == 0 and r["d_cov_lo"] < -0.2 and r["d_cov_hi"] == pytest.approx(0.0, abs=1e-9)
    per = {**per_type(0.012, wc=0.004), **{f"agree_{k}": r[k] for k in bc.GATE_CI_KEYS}, "agree_prec_mean": 0.95}
    assert bc.gate_condition(per) == ("unestimable", {"precision_floor": "pass", "coverage": "pass",
                                                      "precision_drop": "unestimable"})
    assert status("dinov2-reg-large", per, per_type(0.0)) == "gate unestimable"   # cannot win
    unest = per_type(0.012, wc=0.004, prec_ci=(None, None))
    assert bc.gate_condition(unest)[0] == "unestimable"
    assert status("dinov2-reg-large", unest, per_type(0.0)) == "gate unestimable"
    # NaN only in some panorama resamples: every passing Agree of the arm sits in one panorama
    base, arm = runs_at(y, rng, probs, probs)
    for s in probs:
        a, _, right = arm[s]
        only = np.zeros(len(y), bool)
        only[(g == "p0") & ~y] = True
        arm[s] = (a, only, right)
    if not only.any():
        pytest.skip("panorama p0 has no correct row under this seed")
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    assert 0 < r["prec_dropped"] < 150 and r["d_prec_lo"] is not None


def test_the_gate_bootstrap_is_deterministic_under_its_seed():
    y, g, rng = gate_data(n=800)
    base, arm = runs_at(y, rng, {42: 0.7, 43: 0.8, 44: 0.6}, {42: 0.65, 43: 0.82, 44: 0.55})
    a = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=100, seed=5)
    assert a == bc.gate_delta_ci(y, g, PI, arm, base, n_boot=100, seed=5)
    assert a["d_cov_lo"] != bc.gate_delta_ci(y, g, PI, arm, base, n_boot=100, seed=6)["d_cov_lo"]
    none = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=0, seed=5)
    assert none["d_cov"] == a["d_cov"] and none["d_cov_lo"] is None


def with_ci(r, **extra):
    """A ``summarise`` entry that passes conditions 1 and 2, carrying ``gate_delta_ci`` result ``r``."""
    return {**per_type(0.012, wc=0.004), **{f"agree_{k}": r[k] for k in bc.GATE_CI_KEYS},
            "agree_ci_per_seed": r["per_seed"], "agree_ci_seeds": r["seeds"], "agree_ci_n_boot": r["n_boot"],
            "agree_prec_mean": 0.95, **extra}


def test_the_precision_difference_is_right_over_passed_rows():
    """PR #121 review S3: a known precision difference. The baseline passes every row (precision = the prior's
    correct share, 0.6); the arm passes every correct row and a quarter of the incorrect ones."""
    y, g, rng = gate_data()
    ones = np.ones(len(y), bool)
    passed = ~y | (rng.random(len(y)) < 0.25)
    base = {s: (ones, ones, ~y) for s in bc.SEEDS}
    arm = {s: (ones, passed, ~y) for s in bc.SEEDS}
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=50, seed=1)
    w = np.where(y, PI["incorrect"] / y.mean(), PI["correct"] / (1 - y.mean()))
    want = (w * passed * ~y).sum() / (w * passed).sum() - PI["correct"]
    assert r["d_prec"] == pytest.approx(want) and want == pytest.approx(0.6 / 0.7 - 0.6, abs=0.03)
    assert r["per_seed"][0]["base_prec"] == pytest.approx(PI["correct"])


def test_the_interval_is_the_2_5_and_97_5_percentiles():
    """PR #121 review S3: per-seed Δ coverage −0.20, −0.10 and exactly 0. All three draws land on the 0 seed in 1/27
    (3.7%) of replicates, so the 97.5th percentile is 0 and the 2.5th near −0.20; a 25–75 interval would end near
    −0.07. This is also why, with three seeds, the upper end sits at the best seed's difference."""
    y, g, rng = gate_data()
    ones = np.ones(len(y), bool)
    u = {s: rng.random(len(y)) for s in bc.SEEDS}
    base = {s: (ones, u[s] < 0.7, ~y) for s in u}
    arm = {s: (ones, u[s] < 0.7 + d, ~y) for s, d in zip(u, (-0.20, -0.10, 0.0))}
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=400, seed=1)
    assert r["d_cov_hi"] == pytest.approx(0.0, abs=1e-9) and r["d_cov_lo"] < -0.17


def test_the_precision_interval_resamples_seeds_too():
    """Per-seed Δ precision −0.20, −0.10 and exactly 0 (every row passes; the arm calls a share of correct rows
    wrong): the precision interval, like coverage's, reaches the best seed's 0 and the worst seed's −0.20."""
    y, g, rng = gate_data()
    ones = np.ones(len(y), bool)
    u = rng.random(len(y))
    base = {s: (ones, ones, ~y) for s in bc.SEEDS}
    arm = {s: (ones, ones, ~y & (u >= f)) for s, f in zip(bc.SEEDS, (1 / 3, 1 / 6, 0.0))}
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=400, seed=1)
    assert r["d_prec"] == pytest.approx(-0.10, abs=0.02)
    assert r["d_prec_hi"] == pytest.approx(0.0, abs=1e-9) and r["d_prec_lo"] < -0.17


def test_panoramas_not_rows_are_resampled():
    """PR #121 review S3: one seed (so the seed level adds nothing), passes decided per panorama of 20 rows. Resampling
    panoramas must give a much wider interval than treating every row as its own panorama."""
    rng = np.random.default_rng(3)
    n, size = 2000, 20
    y = rng.random(n) < 0.4
    pano = np.arange(n) // size
    a, b = ((rng.random(n // size) < 0.5)[pano] for _ in range(2))
    run = {42: (np.ones(n, bool), a, ~y)}, {42: (np.ones(n, bool), b, ~y)}
    clustered = bc.gate_delta_ci(y, np.array([f"p{i}" for i in pano]), PI, *run, n_boot=400, seed=1)
    rows = bc.gate_delta_ci(y, np.arange(n).astype(str), PI, *run, n_boot=400, seed=1)
    assert clustered["d_cov"] == rows["d_cov"]
    width = {k: r["d_cov_hi"] - r["d_cov_lo"] for k, r in (("clustered", clustered), ("rows", rows))}
    assert width["clustered"] > 2.5 * width["rows"]


def test_the_prior_weights_are_recomputed_on_every_resample():
    """PR #121 final review N1: the arm passes exactly the correct rows and the baseline exactly the incorrect ones,
    every row Agree. Under weights ``pi(y) / resample share(y)`` their coverages are ``pi(correct)`` and
    ``pi(incorrect)`` in every resample, so the interval collapses to the point; weights frozen at the full-sample
    share would move with each resample's class mix and widen it."""
    y, g, _ = gate_data(n=1200)
    ones = np.ones(len(y), bool)
    arm = {s: (ones, ~y, ~y) for s in bc.SEEDS}
    base = {s: (ones, y, ~y) for s in bc.SEEDS}
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=4)
    want = PI["correct"] - PI["incorrect"]
    assert r["d_cov"] == pytest.approx(want, abs=1e-12)
    assert r["d_cov_lo"] == pytest.approx(want, abs=1e-12) and r["d_cov_hi"] == pytest.approx(want, abs=1e-12)
    assert r["d_prec_lo"] == pytest.approx(1.0, abs=1e-12) and r["d_prec_hi"] == pytest.approx(1.0, abs=1e-12)


def test_a_baseline_with_no_gated_agree_on_every_seed_reads_the_precision_clause_as_pass():
    """PR #121 fix round S1, as amended from #120's final review: only when the baseline gates no Agree on EVERY
    shared seed (its precision undefined throughout) and the arm's is defined on every seed does the relative-precision
    clause read as pass, cause named, the 0.92 floor still applying. A baseline undefined on some seeds only, an
    arm-side or a mixed cause stays unestimable, with its cause named."""
    y, g, rng = gate_data(n=1200)
    probs = {42: 0.7, 43: 0.8, 44: 0.6}
    base, arm = runs_at(y, rng, probs, probs)
    none = np.zeros(len(y), bool)
    kept = dict(base)
    for s in probs:
        base[s] = (base[s][0], none, base[s][2])
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    assert r["d_prec_hi"] is None and all(p["base_prec"] is None and p["prec"] is not None for p in r["per_seed"])
    per = with_ci(r)
    res, cl, notes = bc.gate_detail(per)
    assert res == "pass" and cl["precision_drop"] == "pass" and set(notes) == {"precision_drop"}
    assert notes["precision_drop"].startswith("baseline-side: the baseline gates no Agree on every shared seed (42, 43, 44)")
    assert "0.92 floor still applies" in notes["precision_drop"]
    assert bc.gate_detail(with_ci(r, agree_prec_mean=0.91))[1]["precision_floor"] == "fail"   # the floor still bites
    d = bc.decide(summary("dinov2-reg-large", per, per_type(0.0)))
    arm_r = d["arms"]["dinov2-reg-large"]
    assert arm_r["status"] == "winner"
    by = arm_r["condition_3_by_type"]["CurbRamp"]
    assert by["notes"]["precision_drop"].startswith("baseline-side") and by["shared_seeds"] == 3
    assert "seed 44: the baseline gates no Agree, the arm gates some" in by["flags"]
    text = bc.decision_markdown(d, 1)
    assert "baseline-side" in text and "seed 44: the baseline gates no Agree" in text and "3 shared seeds" in text
    # the baseline undefined on seed 44 only: a precision loss on 42 and 43 could hide behind it, so unestimable
    base = dict(kept)
    base[44] = (base[44][0], none, base[44][2])
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    res, cl, notes = bc.gate_detail(with_ci(r))
    assert (res, cl["precision_drop"]) == ("unestimable", "unestimable")
    assert notes["precision_drop"] == ("baseline-side, some seeds: the baseline gates no Agree on seed 44 but not on "
                                       "every shared seed")
    assert bc.decide(summary("dinov2-reg-large", with_ci(r), per_type(0.0)))["arms"]["dinov2-reg-large"]["status"] \
        == "gate unestimable"
    b44 = kept[44]
    # arm-side: unestimable, cause named
    base[44], a44 = b44, arm[44]
    arm[44] = (a44[0], none, a44[2])
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    res, cl, notes = bc.gate_detail(with_ci(r))
    assert (res, cl["precision_drop"]) == ("unestimable", "unestimable")
    assert notes["precision_drop"] == "arm-side: the arm gates no Agree on seed 44"
    d = bc.decide(summary("dinov2-reg-large", with_ci(r), per_type(0.0)))["arms"]["dinov2-reg-large"]
    assert d["status"] == "gate unestimable"
    assert d["condition_3_by_type"]["CurbRamp"]["notes"]["precision_drop"].startswith("arm-side")
    # mixed: the arm on 44, the baseline on 43
    b43 = base[43]
    base[43] = (b43[0], none, b43[2])
    r = bc.gate_delta_ci(y, g, PI, arm, base, n_boot=300, seed=2)
    res, cl, notes = bc.gate_detail(with_ci(r))
    assert res == "unestimable"
    assert notes["precision_drop"] == "mixed: the arm gates no Agree on seed 44 and the baseline on seed 43"
    # every seed defined but under half the replicates kept: the replicates are named
    rep = bc.gate_detail({**per, "agree_d_prec_hi": None, "agree_prec_dropped": 200,
                          "agree_ci_per_seed": [{"seed": 42, "cov": 0.5, "base_cov": 0.5, "prec": 0.9,
                                                 "base_prec": 0.9}]})
    assert rep[0] == "unestimable" and rep[2]["precision_drop"] == "under half of the 300 bootstrap replicates defined (200 dropped)"
    # no interval at all
    assert bc.gate_detail(per_type(prec_ci=(None, None)))[2]["precision_drop"].startswith("no paired interval")
    # the exception is the precision clause's only: a baseline with no Agree verdict on any seed leaves coverage
    # unestimable
    no_cov = [{"seed": s, "cov": 0.5, "base_cov": None, "prec": 0.95, "base_prec": None} for s in bc.SEEDS]
    res, cl, notes = bc.gate_detail({**per_type(cov_ci=(None, None), prec_ci=(None, None)),
                                     "agree_ci_per_seed": no_cov})
    assert (res, cl["coverage"], cl["precision_drop"]) == ("unestimable", "unestimable", "pass")
    assert notes["coverage"] == "baseline-side: the baseline gives no Agree verdict on every shared seed (42, 43, 44)"


def test_margin_and_floor_edges_pass():
    """PR #121 fix round N1: an upper end exactly at the margin, and a precision exactly at the floor, pass."""
    assert bc.gate_condition(per_type(cov=0.47, cov_ci=(-0.05, -0.02)))[1]["coverage"] == "pass"
    assert bc.gate_condition(per_type(prec=0.95, prec_ci=(-0.03, -0.01)))[1]["precision_drop"] == "pass"
    assert bc.gate_condition(per_type(prec=0.92, base_prec=0.92))[1]["precision_floor"] == "pass"


def test_partial_precision_means_and_a_short_seed_count_are_marked():
    """PR #121 fix round (from #120's final review): with a seed's precision undefined the paired mean is undefined,
    so the means over the seeds where both are defined are printed when they differ by more than 0.01; fewer than
    three shared seeds is marked beside condition 3. Neither decides."""
    seeds = [{"seed": 42, "cov": 0.5, "base_cov": 0.5, "prec": 0.93, "base_prec": 0.96},
             {"seed": 43, "cov": 0.5, "base_cov": 0.5, "prec": 0.94, "base_prec": 0.96},
             {"seed": 44, "cov": 0.5, "base_cov": 0.5, "prec": 0.95, "base_prec": None}]
    r = {**per_type(prec_ci=(None, None)), "agree_ci_per_seed": seeds, "agree_ci_seeds": [42, 43, 44]}
    assert "precision over the seeds where both models' is defined (42, 43): arm 0.935, baseline 0.960, Δ -0.025, " \
           "beyond the margin -0.01" in bc.gate_flags(r)
    assert bc.gate_detail(r)[1]["precision_drop"] == "unestimable"
    close = [dict(p, prec=p["base_prec"] or 0.95) for p in seeds]
    assert not any(f.startswith("precision over") for f in bc.gate_flags({**r, "agree_ci_per_seed": close}))
    two = {**per_type(), "agree_ci_per_seed": seeds[:2], "agree_ci_seeds": [42, 43]}
    assert bc.gate_flags(two)[0] == "2 shared seeds (under 3)"
    assert "| 2 (under 3) |" in bc.condition3_markdown(summary("dinov2-reg-large", two, per_type()))
    assert "| 3 |" in bc.condition3_markdown(summary("dinov2-reg-large", r, per_type()))


def test_the_partial_seed_precision_mark_fires_on_a_loss_only():
    """PR #121 final review N3: like #120's ``defined_seed_mark``, the mark fires only when the arm is below the
    baseline by more than 0.01 over the seeds where both are defined; an arm better there by as much is not marked."""
    def flags(arm_prec):
        seeds = [{"seed": 42, "cov": 0.5, "base_cov": 0.5, "prec": arm_prec, "base_prec": 0.95},
                 {"seed": 43, "cov": 0.5, "base_cov": 0.5, "prec": arm_prec, "base_prec": 0.95},
                 {"seed": 44, "cov": 0.5, "base_cov": 0.5, "prec": 0.95, "base_prec": None}]
        return [f for f in bc.gate_flags({"agree_ci_per_seed": seeds}) if f.startswith("precision over")]
    assert flags(0.93) == ["precision over the seeds where both models' is defined (42, 43): arm 0.930, baseline "
                           "0.950, Δ -0.020, beyond the margin -0.01"]
    assert flags(0.97) == []          # better by 0.02: no mark
    assert flags(0.945) == []         # a loss within the margin: no mark


def test_flags_never_decide_and_reach_both_reports():
    """Flags: a seed whose Δ coverage is below −0.06, and on a winner a point estimate beyond the margin whose interval
    spans it. Listed in decision.md and gate.md; the status is unchanged."""
    reg = "dinov2-reg-large"
    seeds = [{"seed": 42, "cov": 0.42, "base_cov": 0.5, "prec": 0.95, "base_prec": 0.95},
             {"seed": 43, "cov": 0.5, "base_cov": 0.5, "prec": 0.95, "base_prec": 0.95},
             {"seed": 44, "cov": 0.49, "base_cov": 0.5, "prec": 0.95, "base_prec": 0.95}]
    cr = {**per_type(0.012, wc=0.004, cov=0.47, cov_ci=(-0.08, 0.01)), "agree_ci_per_seed": seeds,
          "agree_cov_dropped": None, "agree_prec_dropped": None}
    s = summary(reg, cr, per_type(0.0))
    d = bc.decide(s)
    assert d["arms"][reg]["status"] == "winner"
    flags = d["arms"][reg]["condition_3_by_type"]["CurbRamp"]["flags"]
    assert flags == ["seed 42: Δ coverage -0.080, below -0.06",
                     "Δ coverage: point estimate -0.030 beyond the margin -0.02, interval spans it"]
    dec = bc.decision_markdown(d, 1)
    gate = bc.condition3_markdown(s, d)
    for text in (dec, gate):
        assert "seed 42: Δ coverage -0.080, below -0.06" in text and "interval spans it" in text
        assert "never decide" in text
    assert "None" not in gate   # N2: missing dropped counts print blank
    # not a winner: the per-seed flag stays, the winner flag does not
    assert "interval spans it" not in bc.condition3_markdown(s)
    assert bc.gate_flags(cr) == flags[:1]
    # edges: −0.05 on a seed is not below −3 × 0.02; a point estimate beyond the margin with the interval wholly
    # beyond it too (the clause fails) is not "spans it"
    seeds[0]["cov"] = 0.45
    assert bc.gate_flags(cr) == []
    assert bc.winner_flags({"agree_d_cov": -0.05, "agree_d_cov_hi": -0.03}) == []
    assert bc.winner_flags({"agree_d_cov": -0.05, "agree_d_cov_hi": -0.02}) != []


def test_the_precision_floor_reads_every_paired_seed():
    """PR #121 fix round N3: the 0.92 floor is judged on the mean over every paired seed, and is unestimable (cause
    named) when one seed's run gates no Agree."""
    rows, gates = [], []
    for arm in ("dinov2-large", "dinov2-reg-large"):
        for s, p in zip(bc.SEEDS, (0.95, float("nan"), 0.95)):
            rows.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": 1e-5, "d_heldout": 0.01})
            gates.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": 1e-5, "rule": "majority2", "split": "test",
                          "verdict": "correct", "gate_precision": p, "gate_coverage": 0.5})
    r = bc.summarise(rows, gates)["dinov2-reg-large"]["CurbRamp"]
    assert r["agree_prec_mean"] is None
    res, cl, notes = bc.gate_detail(r)
    assert cl["precision_floor"] == "unestimable"
    assert notes["precision_floor"].startswith("arm-side: the arm gates no Agree on seed 43")
    gates[4]["gate_precision"] = 0.86   # the arm's seed 43: mean (0.95 + 0.86 + 0.95) / 3 = 0.92
    r = bc.summarise(rows, gates)["dinov2-reg-large"]["CurbRamp"]
    assert r["agree_prec_mean"] == pytest.approx(0.92)


def test_summarise_carries_the_gate_intervals_into_the_rule():
    rows, gates = [], []
    for arm in ("dinov2-large", "dinov2-reg-large"):
        for s in bc.SEEDS:
            rows.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": 1e-5, "d_heldout": 0.01})
            gates.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": 1e-5, "rule": "majority2", "split": "test",
                          "verdict": "correct", "gate_precision": 0.95, "gate_coverage": 0.5 + (s - 43) / 10})
    ci = {"d_cov": -0.03, "d_cov_lo": -0.09, "d_cov_hi": 0.02, "cov_dropped": 0, "d_prec": 0.0, "d_prec_lo": -0.004,
          "d_prec_hi": 0.005, "prec_dropped": 3, "seeds": [42, 43, 44], "n_boot": 1000}
    r = bc.summarise(rows, gates, {"CurbRamp": {"dinov2-reg-large": ci}})["dinov2-reg-large"]["CurbRamp"]
    assert r["agree_d_cov_hi"] == 0.02 and r["agree_prec_dropped"] == 3 and r["agree_ci_n_boot"] == 1000
    assert r["agree_cov_seeds"] == pytest.approx([0.4, 0.5, 0.6])
    assert bc.gate_condition(r)[0] == "pass"
    bare = bc.summarise(rows, gates)["dinov2-reg-large"]["CurbRamp"]   # no intervals: never passes
    assert bare["agree_d_cov_hi"] is None and bc.gate_condition(bare)[0] == "unestimable"


def test_summarise_reads_the_agree_gate_on_test_under_majority2_at_lr_1e5():
    rows, gates = [], []
    for arm in ("dinov2-large", "dinov2-reg-large"):
        for s in bc.SEEDS:
            rows.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": 1e-5, "d_heldout": 0.01, "heldout_auc": 0.8})
            for rule, split, verdict, lr, p in (("majority2", "test", "correct", 1e-5, 0.95),
                                                ("margin", "test", "correct", 1e-5, 0.11),
                                                ("majority2", "test_heldout", "correct", 1e-5, 0.22),
                                                ("majority2", "test", "incorrect", 1e-5, 0.33),
                                                ("majority2", "test", "correct", 3e-5, 0.44)):
                gates.append({"type": "CurbRamp", "arm": arm, "seed": s, "lr": lr, "rule": rule, "split": split,
                              "verdict": verdict, "gate_precision": p, "gate_coverage": p / 2})
    r = bc.summarise(rows, gates)["dinov2-reg-large"]["CurbRamp"]
    assert r["agree_prec_mean"] == pytest.approx(0.95) and r["agree_cov_mean"] == pytest.approx(0.475)
    assert r["base_agree_prec_mean"] == pytest.approx(0.95) and r["n_seeds"] == 3


def test_several_winners_prefer_the_larger_gain_then_no_custom_code():
    a = summary("dinov2-reg-large", per_type(0.012, wc=0.003), per_type(0.010))
    b = summary("dinov3-vitl16", per_type(0.013, wc=0.003), per_type(0.011))
    assert bc.decide({**a, **b})["recommendation"].startswith("dinov2-reg-large")   # within 0.005: no custom code
    b = summary("dinov3-vitl16", per_type(0.020, wc=0.003), per_type(0.011))
    assert bc.decide({**a, **b})["recommendation"].startswith("dinov3-vitl16")


def test_null_winner_rate_is_seeded_and_falls_with_higher_thresholds():
    a = bc.null_winner_rate(n_sims=4000, seed=1)
    assert a == bc.null_winner_rate(n_sims=4000, seed=1)
    lo = bc.null_winner_rate(thresholds={"CurbRamp": 0.005, "SurfaceProblem": 0.008}, n_sims=4000, seed=1)
    hi = bc.null_winner_rate(thresholds={"CurbRamp": 0.020, "SurfaceProblem": 0.032}, n_sims=4000, seed=1)
    assert lo["expected_winners"] > a["expected_winners"] > hi["expected_winners"]
    assert 0 <= hi["p_any_winner"] <= a["p_any_winner"] <= 1


def test_null_winner_rate_rises_with_the_within_city_correlation():
    """PR #121 review S5: pooled and within-city Δ held-out come from the same rows. rho = 0 is the pre-registered
    (independent) number, unchanged; rho = 1 the bound, about 2.5x higher."""
    r0, r5, r1 = (bc.null_winner_rate(rho=x) for x in (0.0, 0.5, 1.0))
    assert r0 == {**bc.null_winner_rate(), "rho": 0.0}
    assert round(r0["expected_winners"], 3) == 0.021
    assert r0["expected_winners"] < r5["expected_winners"] < r1["expected_winners"]
    assert 0.05 < r1["expected_winners"] < 0.06
    with pytest.raises(ValueError):
        bc.null_winner_rate(rho=1.5)

# ---------------------------------------------------------------------------------------------------------------------
# the within-city difference
# ---------------------------------------------------------------------------------------------------------------------

def test_within_city_delta_ignores_a_per_deployment_offset():
    """An arm that adds a constant per deployment (it only separates deployments) raises the pooled AUC when the
    incorrect share differs by deployment, but its within-city difference is exactly zero."""
    rng = np.random.default_rng(0)
    rows = []
    for c, share, off in (("a", 0.2, 0.0), ("b", 0.5, 0.3), ("c", 0.8, 0.6)):
        for i in range(200):
            y = rng.random() < share
            rows.append({"label_uid": f"{c}:{i}", "city": c, "pano_id": f"{c}{i // 2}", "split": "test_heldout",
                         "true": "incorrect" if y else "correct", "s": 0.3 * y + rng.normal(0, 0.3), "off": off})
    df = pd.DataFrame(rows)
    base = df.assign(p_incorrect=df.s)
    arm = df.assign(p_incorrect=df.s + df.off)
    from analysis import confident_learning as cl
    pooled = cl.overall_deltas({"baseline": base, "arm": arm}, ["arm"], 0)[("arm", "test_heldout")]["delta"]
    w = bc.within_city_delta(arm, base, "test_heldout", n_boot=200, seed=0)
    assert pooled > 0.05
    assert w["delta"] == pytest.approx(0.0, abs=1e-12) and w["n_cities"] == 3
    assert w["delta_lo"] == pytest.approx(0.0, abs=1e-12) and w["delta_hi"] == pytest.approx(0.0, abs=1e-12)
    better = df.assign(p_incorrect=df.s + 0.5 * (df.true == "incorrect"))
    w2 = bc.within_city_delta(better, base, "test_heldout", n_boot=200, seed=0)
    assert w2["delta"] > 0 and 0 < w2["delta_lo"] < w2["delta"] < w2["delta_hi"]   # resampled, not collapsed
    assert w2 == bc.within_city_delta(better, base, "test_heldout", n_boot=200, seed=0)


def test_within_weights_cities_by_n_incorrect_times_n_correct():
    """``_within`` (the bootstrap's within-city AUC) must equal retrain_report.within_city_auc on the same rows; the
    cities differ in size and class balance, so equal weights would give another number."""
    from analysis import retrain_report as rr
    rng = np.random.default_rng(5)
    y, s, city = [], [], []
    for c, n, share, sig in (("a", 300, 0.5, 2.0), ("b", 40, 0.1, 0.0), ("c", 120, 0.3, 0.5)):
        yy = rng.random(n) < share
        y += list(yy)
        s += list(sig * yy + rng.normal(0, 1, n))
        city += [c] * n
    y, s, city = np.array(y), np.array(s), np.array(city)
    groups = [np.flatnonzero(city == c) for c in ("a", "b", "c")]
    wa, wb = bc._within(y, s, -s, None, groups, 10)
    ref, _ = rr.within_city_auc(y, s, pd.Series(city), 10)
    assert wa == pytest.approx(ref) and wb == pytest.approx(1 - ref)
    flat = np.mean([bc.fast_auc(y[g], s[g]) for g in groups])
    assert abs(flat - ref) > 0.02


def test_fast_auc_equals_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = rng.random(300) < 0.4
    s = np.round(rng.random(300), 2)   # ties
    assert bc.fast_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(bc.fast_auc(np.zeros(5, bool), s[:5]))


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic $KEEP and split dir
# ---------------------------------------------------------------------------------------------------------------------

CITIES = {"train": ["amsterdam", "chicago-il", "seattle-wa"], "val": ["amsterdam", "chicago-il", "seattle-wa"],
          "test": ["amsterdam", "chicago-il", "seattle-wa"], "test_heldout": ["pubtown-wa", PRIVATE]}
N = {"train": 80, "val": 120, "test": 180, "test_heldout": 160}


def make_split(rng, label_type):
    rows, i = [], 0
    for sp, n in N.items():
        for k in range(n):
            c = CITIES[sp][k % len(CITIES[sp])]
            cls = "incorrect" if rng.random() < 0.4 else "correct"
            rows.append({"label_uid": f"{c}:{i}", "city": c, "label_id": i, "pano_id": f"p{i // 2}",
                         "label_type": label_type, "label_class": cls, "crop_path": f"{cls}/{c}_{i}.webp",
                         "split": sp, "hard": 0, "severity": 1 + k % 3,
                         "tags": json.dumps(["steep"] if k % 2 else ["narrow", "steep"])})
            i += 1
    return pd.DataFrame(rows)


def make_pred(split, rng, signal):
    ev = split[split.split.isin(["val", "test", "test_heldout"])]
    y = (ev.label_class == "incorrect").to_numpy()
    p = 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(0, 1, len(ev)))))
    return pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                         "p_correct": 1 - p, "p_incorrect": p})


def write_run(keep, lt, tag, split_sha, seed, key, lr, pred, random_init=False, accepts=False):
    d = keep / lt / "runs" / tag
    d.mkdir(parents=True)
    pred.to_csv(d / f"predictions_{tag}.csv", index=False)
    (d / f"metrics_{tag}.json").write_text("{}")
    (d / "run_key.json").write_text(json.dumps({"label_type": lt, "split_sha256": split_sha, "seed": seed}))
    (d / "training_run.json").write_text(json.dumps({
        "seed": seed, "epochs": 25, "learning_rate": lr, "model_accepts_loss_kwargs": accepts,
        "started_at": "2026-11-20T00:00:00Z", "finished_at": "2026-11-20T01:00:00Z", "resumed_from": None,
        "peak_gpu_memory_bytes": 20 * 2**30,
        "backbone": {"key": key, "random_init": random_init, "params": int(bc.bb.get(key).params_m * 1e6),
                     "tokens_per_image": bc.bb.get(key).tokens_per_image, "model_class": bc.bb.MODEL_CLASS[bc.bb.get(key).family],
                     "family": bc.bb.get(key).family}}))
    (d / "trainer_state.json").write_text(json.dumps({"max_steps": 600, "log_history": []}))
    (d / "gpu.txt").write_text("NVIDIA L40S, 46068 MiB\n")
    return d


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    split_dir, keep = tmp_path / "split", tmp_path / "keep"
    inputs = {"files": {}, "split_sha256": {}}
    for lt in ("CurbRamp", "SurfaceProblem"):
        (split_dir / lt).mkdir(parents=True)
        split = make_split(rng, lt)
        split.to_csv(split_dir / lt / "split.csv", index=False)
        s = sha(split_dir / lt / "split.csv")
        inputs["split_sha256"][lt] = s
        for seed, suffix in bc.ROUND3_REF.items():
            make_pred(split, rng, 1.0).to_csv(split_dir / lt / f"predictions{suffix}.csv", index=False)
        for seed in bc.SEEDS:
            write_run(keep, lt, bc.make_tag("bb-", "dinov2-large", seed), s, seed, "dinov2-large", 1e-5,
                      make_pred(split, rng, 1.0))
            write_run(keep, lt, bc.make_tag("bb-", "dinov2-reg-large", seed), s, seed, "dinov2-reg-large", 1e-5,
                      make_pred(split, rng, 1.6))
        write_run(keep, lt, bc.make_tag("bb-", "dinov2-large", 42, 3e-5), s, 42, "dinov2-large", 3e-5,
                  make_pred(split, rng, 1.0))
        # three runs that must be flagged, never used
        write_run(keep, lt, "bb-convnextv2-base", "0" * 64, 42, "convnextv2-base", 1e-5, make_pred(split, rng, 1.0))
        write_run(keep, lt, "bb-convnextv2-base-s43", s, 44, "convnextv2-base", 1e-5, make_pred(split, rng, 1.0))
        write_run(keep, lt, "bb-convnextv2-base-s44", s, 44, "convnextv2-base", 1e-5, make_pred(split, rng, 1.0),
                  random_init=True)
        write_run(keep, lt, "smoke-dinov2-large", s, 42, "dinov2-large", 1e-5, make_pred(split, rng, 1.0))  # other prefix
    (keep / "backbone_jobs.txt").write_text("job 1\n")
    (keep / "tasks_pilot_x.tsv").write_text("CurbRamp\tdinov2-large\t42\tbb-dinov2-large\t1e-5\n")
    inp = tmp_path / "inputs.json"
    inp.write_text(json.dumps(inputs))
    pub = tmp_path / "public.txt"
    pub.write_text("\n".join(PUBLIC) + "\n")
    return {"split_dir": split_dir, "keep": keep, "inputs": inp, "public": pub, "tmp": tmp_path}


def test_consistency_flags_hash_seed_and_random_init(synthetic):
    d = synthetic["keep"] / "CurbRamp" / "runs"
    s = json.loads(synthetic["inputs"].read_text())["split_sha256"]["CurbRamp"]
    ok = bc.consistency(d / "bb-dinov2-reg-large-s43", "bb-dinov2-reg-large-s43",
                        bc.parse_tag("bb-dinov2-reg-large-s43", "bb-"), s)
    assert ok == []
    assert "split_sha256" in bc.consistency(d / "bb-convnextv2-base", "bb-convnextv2-base",
                                            bc.parse_tag("bb-convnextv2-base", "bb-"), s)[0]
    assert "seed 44 != tag's 43" in bc.consistency(d / "bb-convnextv2-base-s43", "bb-convnextv2-base-s43",
                                                   bc.parse_tag("bb-convnextv2-base-s43", "bb-"), s)[0]
    assert any("random_init" in w for w in bc.consistency(d / "bb-convnextv2-base-s44", "bb-convnextv2-base-s44",
                                                          bc.parse_tag("bb-convnextv2-base-s44", "bb-"), s))
    assert bc.consistency(d / "bb-dinov2-reg-large-s43", "bb-dinov2-reg-large-s43",
                          bc.parse_tag("bb-dinov2-reg-large-s43", "bb-"), s, epochs=25) == []
    assert bc.consistency(d / "bb-dinov2-reg-large-s43", "bb-dinov2-reg-large-s43",
                          bc.parse_tag("bb-dinov2-reg-large-s43", "bb-"), s, epochs=1) == ["epochs 25 != 1"]
    for name, edit, reason in (
            ("lr", lambda r: r.update(learning_rate=3e-5), "learning_rate 3e-05 != tag's 1e-05"),
            ("key", lambda r: r["backbone"].update(key="dinov2-large"), "backbone dinov2-large != tag's dinov2-reg-large"),
            ("loss", lambda r: r.update(model_accepts_loss_kwargs=True), "model_accepts_loss_kwargs True"),
            ("versions", lambda r: r.update(versions={"transformers": "4.55.0", "torch": "2.7.1+cu126"}),
             "transformers 4.55.0 != the study env's 4.56.2"),
            ("revision", lambda r: r["backbone"].update(revision="a" * 40, resolved_revision="b" * 40),
             "resolved_revision bbbbbbbbbbbb != the pinned aaaaaaaaaaaa")):
        run_dir = d / "bb-dinov2-reg-large-s44"
        path = run_dir / "training_run.json"
        orig = path.read_text()
        rec = json.loads(orig)
        edit(rec)
        path.write_text(json.dumps(rec))
        got = bc.consistency(run_dir, "bb-dinov2-reg-large-s44", bc.parse_tag("bb-dinov2-reg-large-s44", "bb-"), s)
        path.write_text(orig)
        assert any(w.startswith(reason) for w in got) and len(got) == 1, (name, got)
    # the study env's versions (with a local build suffix) and a matching revision pass
    rec = json.loads((d / "bb-dinov2-reg-large-s44" / "training_run.json").read_text())
    rec["versions"] = {"transformers": "4.56.2", "torch": "2.7.1+cu126", "pandas": "2.3.1", "numpy": "1.26.4"}
    rec["backbone"].update(revision="a" * 40, resolved_revision="a" * 40)
    (d / "bb-dinov2-reg-large-s44" / "training_run.json").write_text(json.dumps(rec))
    assert bc.consistency(d / "bb-dinov2-reg-large-s44", "bb-dinov2-reg-large-s44",
                          bc.parse_tag("bb-dinov2-reg-large-s44", "bb-"), s) == []
    os.remove(d / "bb-dinov2-reg-large" / "run_key.json")
    assert bc.consistency(d / "bb-dinov2-reg-large", "bb-dinov2-reg-large", bc.parse_tag("bb-dinov2-reg-large", "bb-"),
                          s) == ["missing run_key.json"]


def run_report(synthetic, out, *extra):
    return bc.main(["report", "--split-dir", str(synthetic["split_dir"]), "--runs-dir", str(synthetic["keep"]),
                    "--inputs", str(synthetic["inputs"]), "--shares", SHARES, "--public-cities", str(synthetic["public"]),
                    "--out", str(out), "--n-boot", "50", *extra])


EXPECTED = {"arms.md", "arms.json", "seeds.md", "gate.md", "slices.md", "cost.md", "env.md", "lr.md", "rampnet.md",
            "decision.md", "decision.json", "null.md", "provenance.json"}


def test_report_writes_every_file_and_names_no_private_deployment(synthetic, capsys):
    out = synthetic["tmp"] / "out"
    assert run_report(synthetic, out) == 0
    printed = capsys.readouterr().out
    assert set(os.listdir(out)) == EXPECTED
    text = "".join(open(out / n).read() for n in EXPECTED)
    assert PRIVATE not in text and PRIVATE not in printed
    assert "private city A (US)" not in text or "pubtown-wa" in text
    arms = (out / "arms.md").read_text()
    assert "pubtown-wa" in arms and re.search(r"private city A \((US|outside the US)\)", arms)
    # the three flagged runs are listed, never used
    assert arms.count("INCOMPLETE") == 1 and all(t in arms for t in ("`bb-convnextv2-base`", "`bb-convnextv2-base-s43`",
                                                                      "`bb-convnextv2-base-s44`"))
    a = json.loads((out / "arms.json").read_text())
    assert {r["arm"] for r in a["rows"]} == {"dinov2-large", "dinov2-reg-large"}
    reg = [r for r in a["rows"] if r["arm"] == "dinov2-reg-large"]
    assert len(reg) == 6 and all(r["d_heldout"] is not None and r["wc_d_heldout"] is not None for r in reg)
    assert all(r["baseline_tag"] == bc.make_tag("bb-", "dinov2-large", r["seed"]) for r in reg)   # same seed
    s = a["summary"]["dinov2-reg-large"]["CurbRamp"]
    assert s["n_seeds"] == 3 and s["d_heldout_mean"] > 0
    # every output hash in provenance matches, and the inputs it read are hashed
    prov = json.loads((out / "provenance.json").read_text())
    for name, h in prov["outputs"].items():
        assert sha(out / name) == h
    files = prov["files_read"]
    assert {"split-dir/CurbRamp/split.csv", "split-dir/SurfaceProblem/split.csv", "split-dir/CurbRamp/predictions.csv",
            "split-dir/SurfaceProblem/predictions_baseline-s44.csv", "runs-dir/backbone_jobs.txt",
            "runs-dir/CurbRamp/runs/bb-dinov2-reg-large-s43/run_key.json",
            "runs-dir/SurfaceProblem/runs/bb-convnextv2-base/run_key.json"} <= set(files)   # an INCOMPLETE run's too
    assert files["split-dir/SurfaceProblem/split.csv"] == sha(synthetic["split_dir"] / "SurfaceProblem" / "split.csv")
    assert PRIVATE not in json.dumps(prov)


def test_gate_tables_never_mix_rules(synthetic):
    out = synthetic["tmp"] / "out"
    run_report(synthetic, out)
    gate = (out / "gate.md").read_text()
    sections = gate.split("## prior rule ")[1:]
    assert [s.split("`")[1] for s in sections] == ["majority2", "margin"]
    rows = json.loads((out / "decision.json").read_text())
    assert rows["thresholds"] == bc.THRESHOLDS


def test_report_reads_condition_3_from_the_paired_intervals(synthetic):
    out = synthetic["tmp"] / "out"
    run_report(synthetic, out)
    gate = (out / "gate.md").read_text()
    assert "## Condition 3: Agree at the gate on `test` under `majority2`" in gate
    assert gate.index("## Condition 3") < gate.index("## prior rule ")
    a = json.loads((out / "arms.json").read_text())
    checked, undefined = [], []
    for t in bc.PILOT_TYPES:
        s = a["summary"]["dinov2-reg-large"][t]
        assert s["agree_ci_seeds"] == list(bc.SEEDS) and s["agree_ci_n_boot"] == 50
        assert s["agree_d_cov_lo"] is not None and s["agree_d_cov_lo"] <= s["agree_d_cov_hi"]
        # the interval's point estimate is the paired mean of the gate table's own per-seed coverage
        assert s["agree_d_cov"] == pytest.approx(np.mean(s["agree_cov_seeds"]) - np.mean(s["base_agree_cov_seeds"]))
        # and each seed's precision (correct verdicts over passed ones) is the gate table's, for both models; the
        # paired mean is that of the gate table where every seed's is defined, and undefined otherwise
        for per, prec, base_prec in zip(s["agree_ci_per_seed"], s["agree_prec_seeds"], s["base_agree_prec_seeds"]):
            for mine, table in ((per["prec"], prec), (per["base_prec"], base_prec)):
                assert (mine is None) == (table is None)
                if table is not None:
                    assert mine == pytest.approx(table, rel=1e-9)
                    checked.append(table)
        if None in s["agree_prec_seeds"] + s["base_agree_prec_seeds"]:
            assert s["agree_d_prec"] is None
            undefined.append(t)
        else:
            assert s["agree_d_prec"] == pytest.approx(np.mean(s["agree_prec_seeds"])
                                                      - np.mean(s["base_agree_prec_seeds"]), rel=1e-9)
        assert "agree_d_cov" not in a["summary"]["dinov2-large"][t]
    assert any(v < 1 for v in checked)   # a precision under 1 was compared, so a wrong numerator would show
    d = json.loads((out / "decision.json").read_text())["decision"]["arms"]["dinov2-reg-large"]
    assert set(d["condition_3_by_type"]) == set(bc.PILOT_TYPES)
    for t in undefined:   # the fixture's baseline gates no Agree on a seed: the cause is named in every output
        note = d["condition_3_by_type"][t]["notes"]["precision_drop"]
        assert note.startswith("baseline-side") or note.startswith("arm-side") or note.startswith("mixed")
        assert note in gate and note in (out / "decision.md").read_text()
    want = all(bc.gate_condition(a["summary"]["dinov2-reg-large"][t])[0] == "pass" for t in bc.PILOT_TYPES)
    assert d["condition_3"] == want
    assert "nominal 95%" in gate and "| None |" not in gate
    # deterministic: a second report writes the same intervals
    run_report(synthetic, synthetic["tmp"] / "out2")
    b = json.loads((synthetic["tmp"] / "out2" / "arms.json").read_text())
    assert b["summary"]["dinov2-reg-large"] == a["summary"]["dinov2-reg-large"]


def gate_runs(synthetic, lt="CurbRamp"):
    """The synthetic fixture's complete lr 1e-5 runs of ``lt`` as ``gate_intervals`` reads them, with ``pano_id``."""
    split = pd.read_csv(synthetic["split_dir"] / lt / "split.csv", dtype={"pano_id": str, "label_uid": str})
    runs = {}
    for key in ("dinov2-large", "dinov2-reg-large"):
        for seed in bc.SEEDS:
            tag = bc.make_tag("bb-", key, seed)
            pred = pd.read_csv(synthetic["keep"] / lt / "runs" / tag / f"predictions_{tag}.csv", dtype={"label_uid": str})
            pred = pred.merge(split[["label_uid", "pano_id"]], on="label_uid", how="left", validate="one_to_one")
            runs[tag] = {"tag": tag, "status": "complete", "parsed": {"key": key, "seed": seed, "lr": 1e-5},
                         "pred": pred}
    return runs


def _shares():
    with open(SHARES) as f:
        return json.load(f)["types"]


def _same_intervals(a, b):
    assert set(a) == set(b)
    for arm in a:
        for k in bc.GATE_CI_KEYS:
            assert a[arm][k] == pytest.approx(b[arm][k], abs=1e-12), (arm, k)


def test_gate_intervals_align_runs_by_label_uid(synthetic):
    """PR #121 review S3: one run's predictions shuffled; ``_agree_rows`` sorts by ``label_uid``, so the intervals are
    unchanged."""
    runs = gate_runs(synthetic)
    want = bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)
    tag = bc.make_tag("bb-", "dinov2-reg-large", 43)
    runs[tag]["pred"] = runs[tag]["pred"].sample(frac=1, random_state=7).reset_index(drop=True)
    _same_intervals(bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0), want)


def test_gate_intervals_ignore_a_one_class_heldout_split(synthetic):
    """PR #121 fix round S4: only ``test`` is read, so a one-class ``test_heldout`` (whose prior weights are undefined)
    must not stop it."""
    runs = gate_runs(synthetic)
    want = bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)
    for r in runs.values():
        r["pred"].loc[r["pred"]["split"] == "test_heldout", "true"] = "correct"
    _same_intervals(bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0), want)


def test_gate_intervals_refuse_runs_whose_true_classes_differ(synthetic):
    """PR #121 fix round N4: every run must carry the reference run's true classes."""
    runs = gate_runs(synthetic)
    tag = bc.make_tag("bb-", "dinov2-reg-large", 44)
    pred = runs[tag]["pred"]
    i = pred.index[pred["split"] == "test"][0]
    pred.loc[i, "true"] = "incorrect" if pred.loc[i, "true"] == "correct" else "correct"
    with pytest.raises(ValueError, match="true classes differ"):
        bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)


def test_gate_intervals_refuse_runs_whose_rows_differ(synthetic):
    """PR #121 final review N2: every run must carry the reference run's ``label_uid`` rows, a row dropped or a
    ``label_uid`` renamed alike."""
    tag = bc.make_tag("bb-", "dinov2-reg-large", 43)
    runs = gate_runs(synthetic)
    pred = runs[tag]["pred"]
    runs[tag]["pred"] = pred.drop(index=pred.index[pred["split"] == "test"][0])
    with pytest.raises(ValueError, match="rows differ from the baseline's"):
        bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)
    runs = gate_runs(synthetic)
    pred = runs[tag]["pred"]
    pred.loc[pred.index[pred["split"] == "test"][0], "label_uid"] = "elsewhere:999999"
    with pytest.raises(ValueError, match="rows differ from the baseline's"):
        bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)


def test_adding_an_arm_never_moves_another_arms_interval(synthetic):
    """PR #121 final review N2: each arm's generator is seeded from its own name, so a new arm (sorted before the
    existing one) leaves the existing arm's interval exactly as it was."""
    runs = gate_runs(synthetic)
    want = bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)
    assert want["dinov2-reg-large"]["d_cov_lo"] < want["dinov2-reg-large"]["d_cov_hi"]   # not degenerate
    rng = np.random.default_rng(11)
    for seed in bc.SEEDS:
        src = runs[bc.make_tag("bb-", "dinov2-reg-large", seed)]["pred"].copy()
        src["p_incorrect"] = np.clip(src["p_incorrect"] + rng.normal(0, 0.2, len(src)), 0.001, 0.999)
        src["p_correct"] = 1 - src["p_incorrect"]
        tag = bc.make_tag("bb-", "convnextv2-base", seed)
        runs[tag] = {"tag": tag, "status": "complete", "parsed": {"key": "convnextv2-base", "seed": seed, "lr": 1e-5},
                     "pred": src}
    got = bc.gate_intervals("CurbRamp", runs, _shares(), 40, 0)
    assert set(got) == {"convnextv2-base", "dinov2-reg-large"}
    _same_intervals({"dinov2-reg-large": got["dinov2-reg-large"]}, want)


def _curbramp_pi():
    from analysis import prior_rules as pr
    return pr.rule_prior(_shares(), "CurbRamp", bc.DECISION_RULE)


def test_agree_rows_give_a_row_without_a_panorama_its_own_cluster(synthetic):
    """PR #121 final review N6: a missing or empty ``pano_id`` becomes ``uid:<label_uid>`` (as #120 does), never the
    string "nan", so such rows are separate bootstrap clusters."""
    pred = next(iter(gate_runs(synthetic).values()))["pred"]
    test_idx = pred.index[pred["split"] == "test"]
    pred.loc[test_idx[:2], "pano_id"] = np.nan
    pred.loc[test_idx[2], "pano_id"] = ""
    rows = bc._agree_rows(pred, _curbramp_pi())
    lost = set(pred.loc[test_idx[:3], "label_uid"].astype(str))
    got = rows.loc[rows["label_uid"].isin(lost)].set_index("label_uid")["pano_id"].to_dict()
    assert got == {u: f"uid:{u}" for u in lost}
    assert not rows["pano_id"].isin(["nan", "", "<NA>", "None"]).any()
    kept = pred.loc[test_idx[3:]].set_index("label_uid")["pano_id"].astype(str)
    assert rows.set_index("label_uid").loc[kept.index, "pano_id"].tolist() == kept.tolist()


def test_agree_rows_silence_only_the_one_class_warning(synthetic, monkeypatch):
    """PR #121 final review N5: the one-class message is silenced; any other warning surfaces."""
    import warnings
    from analysis import prior_rules as pr
    real = pr.served_and_weights

    def noisy(pred, pi):
        warnings.warn("evaluation on 'test_heldout' skipped: no examples of class ['incorrect'] to reweight")
        warnings.warn("something else entirely")
        return real(pred, pi)
    monkeypatch.setattr(pr, "served_and_weights", noisy)
    pred = next(iter(gate_runs(synthetic).values()))["pred"]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        bc._agree_rows(pred, _curbramp_pi())
    assert [str(w.message) for w in seen] == ["something else entirely"]


def test_slices_and_lr_and_env_tables(synthetic):
    out = synthetic["tmp"] / "out"
    run_report(synthetic, out)
    sl = (out / "slices.md").read_text()
    assert "by severity" in sl and "by tag" in sl and "training rows" in sl and "round-3 AUC" in sl
    assert "steep" in sl and "narrow" in sl
    assert "`dinov2-large`" in (out / "lr.md").read_text()
    env = (out / "env.md").read_text()
    assert env.count("| CurbRamp |") == 3 and env.count("| SurfaceProblem |") == 3


def test_load_runs_flags_a_short_or_non_finite_prediction_file(synthetic):
    lt, d = "CurbRamp", synthetic["keep"] / "CurbRamp" / "runs"
    split = pd.read_csv(synthetic["split_dir"] / lt / "split.csv", dtype={"pano_id": str, "label_uid": str})
    s = json.loads(synthetic["inputs"].read_text())["split_sha256"][lt]
    p43, p44 = d / "bb-dinov2-reg-large-s43" / "predictions_bb-dinov2-reg-large-s43.csv", \
        d / "bb-dinov2-reg-large-s44" / "predictions_bb-dinov2-reg-large-s44.csv"
    pd.read_csv(p43).iloc[:-1].to_csv(p43, index=False)
    bad = pd.read_csv(p44)
    bad.loc[3, "p_incorrect"] = np.nan
    bad.to_csv(p44, index=False)
    runs = bc.load_runs(str(synthetic["keep"]), lt, "bb-", split, s, bc.Tracker())
    assert runs["bb-dinov2-reg-large-s43"]["status"] == "INCOMPLETE"
    assert "459 predictions, the split has 460 evaluation rows" in runs["bb-dinov2-reg-large-s43"]["why"]
    assert runs["bb-dinov2-reg-large-s44"]["why"] == ["non-finite scores"]
    assert runs["bb-dinov2-reg-large"]["status"] == "complete"


def test_report_holds_while_submitted_rows_are_incomplete_and_names_arms_not_run(synthetic):
    out = synthetic["tmp"] / "out"
    rows_ = "".join(f"{lt}\tdinov3-vitl16\t{s}\t{bc.make_tag('bb-', 'dinov3-vitl16', s)}\t1e-5\n"
                    for s in bc.SEEDS for lt in bc.PILOT_TYPES)
    (synthetic["keep"] / "tasks_dinov3_x.tsv").write_text(rows_)
    run_report(synthetic, out)
    d = json.loads((out / "decision.json").read_text())["decision"]
    assert d["recommendation"].startswith("held: dinov3-vitl16") and d["choice"] is None
    assert "rampnet-convnextv2-base" in d["not_run"]
    md = (out / "decision.md").read_text()
    assert "Submitted but incomplete" in md and "`dinov3-vitl16` (CurbRamp seed 42" in md
    run_report(synthetic, out, "--drop-arms", "dinov3-vitl16")
    d = json.loads((out / "decision.json").read_text())["decision"]
    assert not d["recommendation"].startswith("held") and d["arms"]["dinov3-vitl16"]["status"] == "dropped"
    null = (out / "null.md").read_text()
    assert "| 0 |" in null and "| 0.5 |" in null and "| 1 |" in null and "the range 0.021 to 0.05" in null
    with pytest.raises(SystemExit, match="not registry keys"):
        run_report(synthetic, out, "--drop-arms", "nope")


def test_report_refuses_the_other_three_types(synthetic):
    with pytest.raises(SystemExit, match="follow-up PR"):
        run_report(synthetic, synthetic["tmp"] / "out", "--label-types", "CurbRamp,Obstacle")


def test_report_refuses_a_split_that_is_not_the_pinned_one(synthetic):
    p = synthetic["split_dir"] / "CurbRamp" / "split.csv"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(SystemExit, match="not the pinned"):
        run_report(synthetic, synthetic["tmp"] / "out")


def test_the_smoke_prefix_reads_only_smoke_runs(synthetic):
    out = synthetic["tmp"] / "smoke"
    run_report(synthetic, out, "--tag-prefix", "smoke-", "--label-types", "CurbRamp")
    a = json.loads((out / "arms.json").read_text())
    assert [r["tag"] for r in a["rows"]] == ["smoke-dinov2-large"]


def test_every_label_merge_is_validated():
    src = open(os.path.join(REPO, "analysis", "backbone_comparison.py")).read()
    merges = re.findall(r"\.merge\((.*?)\)\s*$", src, flags=re.M | re.S)
    calls = [m.start() for m in re.finditer(r"\.merge\(", src)]
    assert calls
    for start in calls:
        call = src[start:src.index(")", src.index("validate", start) if "validate" in src[start:start + 300] else start)]
        assert "validate=" in src[start:start + 300], src[start:start + 120]
    assert merges is not None


# ---------------------------------------------------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------------------------------------------------

def make_round3(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "r3"
    files = {}
    for lt in ("CurbRamp",):
        (src / lt / "correct").mkdir(parents=True)
        (src / lt / "incorrect").mkdir()
        split = make_split(rng, lt)
        big = pd.concat([split.assign(label_uid=split.label_uid + f"-{k}", split=split.split) for k in range(4)])
        big.to_csv(src / lt / "split.csv", index=False)
        for name in ("manifest.csv", "generation_run.json", "predictions.csv", "metrics.json"):
            (src / lt / name).write_text(f"{lt} {name}\n")
        for name in ("split.csv", "manifest.csv", "generation_run.json", "predictions.csv", "metrics.json"):
            files[f"{lt}/{name}"] = sha(src / lt / name)
    (src / "deployment_priors.json").write_text("{}")
    files["deployment_priors.json"] = sha(src / "deployment_priors.json")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"files": files, "split_sha256": {"CurbRamp": files["CurbRamp/split.csv"]}}))
    return src, inputs


def listing(root):
    return sorted((d, tuple(sorted(f)), os.path.getmtime(d)) for d, _, f in os.walk(root))


def test_view_copies_small_files_links_only_crop_dirs_and_never_writes_under_the_source(tmp_path):
    src, inputs = make_round3(tmp_path)
    before = listing(src)
    dst = tmp_path / "view"
    assert bc.main(["view", "--src", str(src), "--dst", str(dst), "--inputs", str(inputs), "--label-types", "CurbRamp"]) == 0
    links = bc._all_links(str(dst))
    assert sorted(os.path.basename(p) for p in links) == ["correct", "incorrect"]
    for name in ("split.csv", "manifest.csv", "generation_run.json", "predictions.csv", "metrics.json"):
        p = dst / "CurbRamp" / name
        assert p.is_file() and not p.is_symlink()
    assert not (dst / "deployment_priors.json").is_symlink()
    # evaluate.py writing into the view lands in the view
    (dst / "CurbRamp" / "predictions_bb-x.csv").write_text("x")
    assert listing(src) == before
    # rebuilding is idempotent
    assert bc.main(["view", "--src", str(src), "--dst", str(dst), "--inputs", str(inputs), "--label-types", "CurbRamp"]) == 0


def test_view_refuses_a_hash_mismatch_and_a_symlinked_file(tmp_path):
    src, inputs = make_round3(tmp_path)
    (src / "CurbRamp" / "metrics.json").write_text("changed\n")
    with pytest.raises(ValueError, match="pinned"):
        bc.main(["view", "--src", str(src), "--dst", str(tmp_path / "v"), "--inputs", str(inputs),
                 "--label-types", "CurbRamp"])
    src2, inputs2 = make_round3(tmp_path / "b")
    dst = tmp_path / "v2"
    (dst / "CurbRamp").mkdir(parents=True)
    os.symlink(src2 / "CurbRamp" / "split.csv", dst / "CurbRamp" / "split.csv")
    with pytest.raises(ValueError, match="symlink"):
        bc.main(["view", "--src", str(src2), "--dst", str(dst), "--inputs", str(inputs2), "--label-types", "CurbRamp"])
    with pytest.raises(SystemExit, match="inside --src"):
        bc.main(["view", "--src", str(src2), "--dst", str(src2 / "v"), "--inputs", str(inputs2)])


def test_smoke_view_is_seeded_balanced_and_pinned(tmp_path):
    src, inputs = make_round3(tmp_path)
    full = pd.read_csv(src / "CurbRamp" / "split.csv")
    counts = {"train": 40, "val": 40, "test": 40, "test_heldout": 40}
    a = bc.smoke_subsample(full, counts, seed=0)
    assert a.equals(bc.smoke_subsample(full, counts, seed=0))
    assert not a.equals(bc.smoke_subsample(full, counts, seed=1))
    assert (a.groupby(["split", "label_class"]).size() == 20).all()
    out = tmp_path / "smoke.json"
    dst = tmp_path / "smoke" / "data"
    big = pd.concat([full.assign(label_uid=full.label_uid + f"-{k}") for k in range(3)])  # enough rows for 320/160
    big.to_csv(src / "CurbRamp" / "split.csv", index=False)
    pins = json.loads(inputs.read_text())
    pins["files"]["CurbRamp/split.csv"] = sha(src / "CurbRamp" / "split.csv")
    inputs.write_text(json.dumps(pins))
    assert bc.main(["view", "--src", str(src), "--dst", str(dst), "--inputs", str(inputs), "--smoke",
                    "--smoke-inputs-out", str(out), "--scratch", str(tmp_path / "scratch")]) == 0
    sub = pd.read_csv(dst / "CurbRamp" / "split.csv")
    assert sub.split.value_counts().to_dict() == {"train": 320, "val": 160, "test": 160, "test_heldout": 160}
    assert (sub.groupby(["split", "label_class"]).size() * 2 == sub.split.value_counts().reindex(
        sub.groupby(["split", "label_class"]).size().index.get_level_values(0)).to_numpy()).all()
    pinned = json.loads(out.read_text())
    assert pinned["split_sha256"]["CurbRamp"] == sha(dst / "CurbRamp" / "split.csv")
    assert sorted(os.path.basename(p) for p in bc._all_links(str(dst))) == ["correct", "incorrect"]
