"""analysis/context_arms.py, the pre-registered decision (issue #14): the joint within-deployment bootstrap, the
seed-mean interval with seed variance, the verdicts (WIN, NO GAIN, REGRESSION, INCOMPLETE), the report-only gate
block, the drift row, the operating characteristics, and the report on the pilot's state (five baseline seeds and
the side arms, no confirmatory arm yet). CPU only, pandas/numpy/scipy/sklearn."""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from analysis import context_arms as ca

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "context-arms-2026-11.md")
HELDOUT = ("gamma", "delta", "zeta-wa")   # gamma and zeta-wa are private in these fixtures
PUBLIC = ("alpha", "beta", "delta")


def make_dataset(root, n_panos=420, seed=0):
    """Toy/split.csv with train/val/test rows in alpha and beta and three held-out deployments."""
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for p in range(n_panos):
        city = ["alpha", "beta", "alpha", "beta", "gamma", "delta", "zeta-wa"][p % 7]
        split = "test_heldout" if city in HELDOUT else ["train", "train", "val", "test"][(p // 7) % 4]
        for _ in range(int(rng.integers(1, 3))):
            cls = "incorrect" if rng.random() < 0.4 else "correct"
            rows.append({"label_uid": f"{city}:{i}", "city": city, "label_id": str(i), "pano_id": f"p{p:04d}",
                         "label_class": cls, "class_source": "margin", "author": "human",
                         "severity": str(1 + i % 3), "tags": '["narrow"]' if i % 3 == 0 else "[]",
                         "crop_path": f"{cls}/{city}_{i}.webp", "split": split, "hard": "0"})
            i += 1
    df = pd.DataFrame(rows)
    base = root / "Toy"
    base.mkdir(parents=True, exist_ok=True)
    df.to_csv(base / "split.csv", index=False, lineterminator="\n")
    (root / "deployment_priors.json").write_text(json.dumps({"Toy": {"correct": 900, "incorrect": 100}}))
    shares = root / "shares.json"
    shares.write_text(json.dumps({"types": {"Toy": {"rules": {"majority2": {"correct": 85, "incorrect": 15},
                                                              "margin": {"correct": 90, "incorrect": 10}}}}}))
    (root / "public.txt").write_text("\n".join(PUBLIC) + "\n")
    return df


METRICS = {"test": {"auc": 0.9, "per_class": {"correct": {"f1-score": 0.8}, "incorrect": {"f1-score": 0.6}}},
           "test_heldout": {"auc": 0.8}}


def write_run(root, df, tag, seed, strength, record=True, record_seed=None, noise_seed=None):
    """predictions_<tag>.csv / metrics_<tag>.json and, for this chain's runs, runs/<tag>/ with a matching key."""
    base = root / "Toy"
    ev = df[df.split != "train"]
    rng = np.random.default_rng(1000 + (seed if noise_seed is None else noise_seed))
    y = (ev.label_class == "incorrect").to_numpy(float)
    p = np.clip(0.5 + strength * (y - 0.5) + rng.normal(0, 0.22, len(ev)), 0.001, 0.999)
    suffix = "" if tag == "baseline" else f"_{tag}"
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class, "p_correct": 1 - p,
                  "p_incorrect": p}).to_csv(base / f"predictions{suffix}.csv", index=False)
    (base / f"metrics{suffix}.json").write_text(json.dumps(METRICS))
    if record:
        rec = base / "runs" / tag
        rec.mkdir(parents=True, exist_ok=True)
        split_sha = ca.sha256_file(str(base / "split.csv"))
        (rec / "run_key.json").write_text(json.dumps({"seed": record_seed or seed, "split_sha256": split_sha}))
        (rec / "training_run.json").write_text(json.dumps({"seed": record_seed or seed}))


def tag_of(variant, seed):
    return variant if seed == 42 else f"{variant}-s{seed}"


def pilot_state(root, df, base_strength=0.30):
    """October's trio, the baseline rerun at five seeds, both side arms at seed 42: what the pilot leaves."""
    for s in (42, 43, 44):
        write_run(root, df, "baseline" if s == 42 else f"baseline-s{s}", s, base_strength, record=False,
                  noise_seed=s + 50)
    for s in (42, 43, 44, 45, 46):
        write_run(root, df, tag_of("baseline-rerun", s), s, base_strength)
    write_run(root, df, "side_info", 42, base_strength + 0.05, noise_seed=7)
    write_run(root, df, "side_info_nohq", 42, base_strength + 0.02, noise_seed=8)


def decide(root, df, arms=("two_scale", "view_heatmap", "side_info", "side_info_nohq"), subset=False, shares=None,
           n_boot=60):
    shares = json.load(open(shares or root / "shares.json"))
    priors = json.load(open(root / "deployment_priors.json"))
    split = pd.read_csv(root / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    return ca.decide_type(str(root), "Toy", split, list(arms), set(PUBLIC), shares, priors["Toy"], subset,
                          "baseline-rerun", n_boot, 0)


def row(d, arm):
    return next(r for r in d["rows"] if r["arm"] == arm)


@pytest.fixture
def pilot(tmp_path):
    df = make_dataset(tmp_path)
    pilot_state(tmp_path, df)
    return tmp_path, df


# ---------------------------------------------------------------------------------------------------- bootstrap

def preds_frame(df, scores, split):
    ev = df[df.split == split]
    return {n: pd.DataFrame({"label_uid": ev.label_uid.values, "split": split, "city": ev.city.values,
                             "pano_id": ev.pano_id.values, "true": ev.label_class.values, "p_incorrect": s})
            for n, s in scores.items()}


def test_within_city_boot_is_paired_and_matches_retrain_report(tmp_path):
    from analysis import retrain_report as rr
    df = make_dataset(tmp_path)
    ev = df[df.split == "test_heldout"]
    rng = np.random.default_rng(2)
    y = (ev.label_class == "incorrect").to_numpy(float)
    scores = {"a": y + rng.normal(0, 0.8, len(ev)), "b": y + rng.normal(0, 0.8, len(ev)), "c": rng.random(len(ev))}
    preds = preds_frame(df, scores, "test_heldout")
    b = ca.within_city_boot(preds, "test_heldout", ["a", "b", "c"], min_n=10, n_boot=200, seed=1)
    for n in ("a", "b", "c"):
        want, k = rr.within_city_auc(y, scores[n], ev.city.values, 10)
        assert b["point"][n] == pytest.approx(want) and k == 3
    assert b["boot"].shape == (200 - b["dropped"], 3) and sorted(b["deployments"]) == sorted(HELDOUT)
    # the same resample for every run: a run that is an exact copy of another has identical bootstrap columns
    b2 = ca.within_city_boot({**preds, "a2": preds["a"]}, "test_heldout", ["a", "a2"], 10, 50, 1)
    assert np.array_equal(b2["boot"][:, 0], b2["boot"][:, 1])
    # deterministic in the seed, and the per-deployment AUCs are reported
    assert np.array_equal(ca.within_city_boot(preds, "test_heldout", ["a"], 10, 30, 5)["boot"],
                          ca.within_city_boot(preds, "test_heldout", ["a"], 10, 30, 5)["boot"])
    assert set(b["per_deployment"]) == set(HELDOUT)


def test_a_small_deployment_is_redrawn_and_counted():
    rows = []
    for k in range(200):   # a big deployment, and a 10-label one with a single incorrect label
        rows.append(("big", f"b{k}", "incorrect" if k % 2 else "correct"))
    for k in range(10):
        rows.append(("small", f"s{k}", "incorrect" if k == 0 else "correct"))
    d = pd.DataFrame(rows, columns=["city", "pano_id", "true"])
    d["label_uid"] = [f"{c}:{i}" for i, c in enumerate(d.city)]
    d["split"] = "test_heldout"
    rng = np.random.default_rng(0)
    d["p_incorrect"] = (d.true == "incorrect") + rng.normal(0, 0.5, len(d))
    b = ca.within_city_boot({"m": d}, "test_heldout", ["m"], min_n=10, n_boot=300, seed=0, max_redraws=3)
    assert b["redraws"]["small"] > 0 and b["redraws"]["big"] == 0
    assert b["dropped"] > 0 and len(b["boot"]) == 300 - b["dropped"]


def test_pooled_and_within_city_can_disagree():
    """Model A separates the deployments (one is mostly incorrect) but ranks nothing within either; model B ranks
    within each deployment. Pooled AUC prefers A, within-city AUC prefers B: the endpoint matters."""
    rng = np.random.default_rng(0)
    rows = []
    for city, share in (("x", 0.8), ("y", 0.2)):
        for k in range(300):
            rows.append((city, f"{city}{k}", "incorrect" if rng.random() < share else "correct"))
    d = pd.DataFrame(rows, columns=["city", "pano_id", "true"])
    d["label_uid"] = [f"{c}:{i}" for i, c in enumerate(d.city)]
    d["split"] = "test_heldout"
    y = (d.true == "incorrect").to_numpy(float)
    a = np.where(d.city == "x", 0.9, 0.1) + rng.normal(0, 0.01, len(d))
    bscore = 0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.15, len(d))
    preds = {"A": d.assign(p_incorrect=a), "B": d.assign(p_incorrect=bscore)}
    w = ca.within_city_boot(preds, "test_heldout", ["A", "B"], 10, 0, 0)["point"]
    pooled = ca._rank_auc(y, np.c_[a, bscore])
    assert pooled[0] > pooled[1] and w["A"] < w["B"]


# ---------------------------------------------------------------------------------------------------- rule

def test_rule_constants_equal_the_docs_json_block():
    text = open(DOC).read()
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    assert any(json.loads(b) == ca.DECISION_RULE for b in blocks), "the doc's fenced json block != DECISION_RULE"


def test_seed_mean_delta_includes_seed_variance():
    narrow = ca.seed_mean_delta([0.90, 0.90, 0.90], [0.88] * 5, var_boot=1e-5)
    wide = ca.seed_mean_delta([0.86, 0.90, 0.94], [0.84, 0.86, 0.88, 0.90, 0.92], var_boot=1e-5)
    assert narrow["delta"] == pytest.approx(wide["delta"]) == pytest.approx(0.02)
    assert (wide["hi"] - wide["lo"]) > 5 * (narrow["hi"] - narrow["lo"] if narrow["lo"] is not None else 0)
    single = ca.seed_mean_delta([0.95], [0.88, 0.89, 0.90], 1e-5)
    assert single["single_arm_seed"] and single["nu"] == pytest.approx(2.0)
    assert ca.seed_mean_delta([], [0.9])["delta"] is None


def test_null_arm_from_the_baselines_own_spread_rarely_regresses():
    oc = ca.operating_characteristics(0.024, 0.01, n_arm=3, n_base=5, n_sim=4000, seed=3)
    assert oc["null_regression"] < 0.05 and oc["false_win"] < 0.05
    assert oc["power"]["0.05"] > oc["power"]["0.03"] > oc["power"]["0.02"]
    five = ca.operating_characteristics(0.024, 0.01, n_arm=5, n_base=5, n_sim=4000, seed=3)
    assert five["power"]["0.03"] > oc["power"]["0.03"]


# ---------------------------------------------------------------------------------------------------- verdicts

def add_arm(root, df, arm, strength, seeds=(42, 43, 44)):
    for s in seeds:
        write_run(root, df, tag_of(arm, s), s, strength, noise_seed=s + 200)


def test_win(pilot):
    root, df = pilot
    add_arm(root, df, "two_scale", 0.9)
    r = row(decide(root, df), "two_scale")
    assert r["verdict"] == "WIN", r
    assert all(c["pass"] for c in r["criteria"][:5]) and r["heldout"]["n_arm"] == 3 and r["heldout"]["n_base"] == 5
    assert set(r["deployments"]) == {"delta", "private city A (outside the US)", "private city B (US)"}


def test_no_gain_is_worded_with_the_power(pilot):
    root, df = pilot
    add_arm(root, df, "view_heatmap", 0.30)
    r = row(decide(root, df), "view_heatmap")
    assert r["verdict"] == "NO GAIN" and r["reason"] == "no gain detected at this power"
    assert any(not c["pass"] for c in r["criteria"][:5])


def test_regression(pilot):
    root, df = pilot
    add_arm(root, df, "two_scale", -0.2)
    assert row(decide(root, df), "two_scale")["verdict"] == "REGRESSION"


@pytest.mark.parametrize("break_it, reason", [
    ("missing_seed", "missing arm seed(s) [44]"),
    ("missing_baseline_seed", "missing baseline-rerun seed(s) [46]"),
    ("record_seed", "record seed 99 for tag seed 43"),
    ("no_record", "run record missing"),
    ("no_gate", "gate block missing"),
    ("subset", "subset shadow"),
])
def test_incomplete(pilot, break_it, reason):
    root, df = pilot
    add_arm(root, df, "two_scale", 0.9)
    shares = None
    subset = False
    if break_it == "missing_seed":
        os.remove(root / "Toy" / "metrics_two_scale-s44.json")
    elif break_it == "missing_baseline_seed":
        os.remove(root / "Toy" / "metrics_baseline-rerun-s46.json")
    elif break_it == "record_seed":
        write_run(root, df, "two_scale-s43", 43, 0.9, record_seed=99, noise_seed=243)
    elif break_it == "no_record":
        import shutil
        shutil.rmtree(root / "Toy" / "runs" / "two_scale")
    elif break_it == "no_gate":
        shares = root / "other_shares.json"
        shares.write_text(json.dumps({"types": {}}))
    elif break_it == "subset":
        subset = True
    r = row(decide(root, df, shares=shares, subset=subset), "two_scale")
    assert r["verdict"] == "INCOMPLETE" and reason in r["reason"], r["reason"]


def test_a_regression_on_one_type_vetoes_only_that_type(tmp_path):
    """Types are separate models: the verdict is per type, nothing carries over."""
    out = {}
    for lt_root, strength in (("one", 0.9), ("two", -0.2)):
        root = tmp_path / lt_root
        df = make_dataset(root)
        pilot_state(root, df)
        add_arm(root, df, "two_scale", strength)
        out[lt_root] = row(decide(root, df), "two_scale")["verdict"]
    assert out == {"one": "WIN", "two": "REGRESSION"}


# ---------------------------------------------------------------------------------------------------- report

def run_report(root, *types):
    out = root / "doc"
    assert ca.main(["report", "--data-dir", str(root), "--label-types", ",".join(types or ("Toy",)),
                    "--public-cities", str(root / "public.txt"), "--n-boot", "20", "--decision-boot", "40",
                    "--slice-min-n", "5", "--shares", str(root / "shares.json"), "--out", str(out)]) == 0
    return out


def test_report_runs_to_the_end_on_the_pilots_state(pilot):
    root, df = pilot
    out = run_report(root)
    for f in ("decision.md", "decision.json", "gate.md", "slices_seeds.md", "arms.md", "provenance.json"):
        assert (out / f).exists(), f
    d = json.load(open(out / "decision.json"))
    assert d["rule"] == ca.DECISION_RULE
    rows = {r["arm"]: r for r in d["types"]["Toy"]["rows"]}
    for arm in ("two_scale", "view_heatmap"):          # confirmatory, not run yet
        assert rows[arm]["verdict"] == "INCOMPLETE" and rows[arm]["reason"] == "not run"
    for arm in ("side_info", "side_info_nohq"):        # exploratory, filled in
        assert rows[arm]["verdict"] == "exploratory" and rows[arm]["heldout"]["delta"] is not None
        assert rows[arm]["heldout"]["single_arm_seed"] and rows[arm]["heldout"]["n_base"] == 5
    assert rows["view_heatmap-r448"]["reason"] == "not run"
    drift = d["types"]["Toy"]["drift"]
    assert drift["rerun_seeds"] == [42, 43, 44, 45, 46] and drift["october_seeds"] == [42, 43, 44]
    assert drift["heldout"]["difference"] is not None and drift["heldout"]["october_sd"] is not None
    oc = d["types"]["Toy"]["oc"]
    assert oc["sd_seed_source"] == "baseline-rerun, 5 seeds" and oc["arm_seeds_5"]["n_arm"] == 5
    md = (out / "decision.md").read_text()
    assert "| two_scale | confirmatory | — | INCOMPLETE |" in md and "Drift row" in md and "80% power at" in md
    gate = (out / "gate.md").read_text()
    assert "| baseline-rerun-s45 | majority2 | test | Agree |" in gate and "| baseline | margin |" in gate
    assert "Wilson lo" in gate
    sl = (out / "slices_seeds.md").read_text()
    assert "training rows" in sl and "side_info_nohq Δ mean" in sl and "round-3 (Oct) AUC" in sl
    prov = json.load(open(out / "provenance.json"))
    assert prov["decision_rule"] == ca.DECISION_RULE and prov["shares_sha256"] and "git_commit" in prov
    r45 = prov["runs"]["Toy/baseline-rerun-s45"]
    assert r45["status"] == "ok" and r45["seed"] == 45 and r45["problem"] is None
    assert prov["runs"]["Toy/baseline"]["status"] == "missing" and prov["runs"]["Toy/baseline"]["problem"] is None


def test_private_heldout_deployments_are_never_named(pilot):
    root, df = pilot
    add_arm(root, df, "two_scale", 0.9)
    out = run_report(root)
    for f in os.listdir(out):
        text = (out / f).read_text()
        assert "gamma" not in text and "zeta-wa" not in text, f
    assert "private city A (outside the US)" in (out / "decision.md").read_text()


def test_report_on_a_subset_shadow_is_never_decidable(pilot):
    root, df = pilot
    add_arm(root, df, "two_scale", 0.9)
    (root / "shadow_run.json").write_text(json.dumps({"subset": True, "types": {}}))
    d = json.load(open(run_report(root) / "decision.json"))
    assert d["subset_shadow"] is True
    assert {r["arm"]: r["verdict"] for r in d["types"]["Toy"]["rows"]}["two_scale"] == "INCOMPLETE"


def test_power_table_from_octobers_trio(pilot):
    root, _ = pilot
    t = ca.power_table(str(root), "Toy", "baseline", n_boot=40, n_sim=500)
    assert t["seeds"] == [42, 43, 44] and t["sd_seed"] > 0 and t["sd_boot"] > 0
    assert t["arm_seeds_5"]["power"]["0.05"] >= t["arm_seeds_3"]["power"]["0.05"] - 0.05
    md = ca.power_markdown({"Toy": t})
    assert md.count("| Toy |") == 2 and "recommended arm seeds" in md
    assert "error" in ca.power_table(str(root), "Toy", "two_scale", n_boot=10)
