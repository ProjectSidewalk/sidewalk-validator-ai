"""analysis/split_leak.py (issue #3): the coupled label-level split, the exposure census, the contrast and its
pooling, the Stage 0 bound, the pre-registered rule, and what may be written for docs/."""
import doctest
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import split_leak as sl  # noqa: E402
import split_data  # noqa: E402
from test_split_data import M_PER_DEG_LAT, write_manifest, write_rawlabels  # noqa: E402

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "split-leak-2026-11.md")
CITIES = ("alpha-wa", "beta-or", "gamma")   # write_manifest's deployments; gamma is held out below
T = "Toy"


def test_doctests():
    assert doctest.testmod(sl).failed == 0


# ---------------------------------------------------------------------------------------------------------------------
# A toy pipeline root: data/<T>/{manifest,split}.csv, data/rawlabels/, and a runs directory beside it
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def root(tmp_path):
    data = tmp_path / "data"
    base = data / T
    base.mkdir(parents=True)
    manifest = write_manifest(base, n_panos=240)
    write_rawlabels(data, manifest)
    assert split_data.main(["--label_type", T, "--out", str(data), "--rawlabels", str(data / "rawlabels"),
                            "--holdout-cities", "gamma"]) == 0
    return {"data": data, "runs": tmp_path / "runs", "raw": data / "rawlabels", "out": tmp_path / "report",
            "split": base / "split.csv", "manifest": manifest}


def make(root, seed=42, **kw):
    return sl.make_split(root["manifest"], root["split"], root["raw"], sl.leaky_split_path(root["runs"], T, seed),
                         seed, **kw)


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------------------------------------------------
# make-split
# ---------------------------------------------------------------------------------------------------------------------

def test_make_split_stops_when_the_recomputed_split_differs(root):
    df = read(root["split"])
    i = df.index[df.split == "train"][0]
    df.loc[i, "split"] = "test"
    df.to_csv(root["split"], index=False, lineterminator="\n")
    with pytest.raises(sl.SplitLeakError, match="does not reproduce"):
        make(root)
    assert not os.path.exists(sl.leaky_split_path(root["runs"], T, 42))


def test_make_split_never_writes_under_the_data_directory(root):
    inside = root["data"] / T / "leaky.csv"
    with pytest.raises(sl.SplitLeakError, match="data directory"):
        sl.make_split(root["manifest"], root["split"], root["raw"], inside, 42)
    assert not inside.exists()
    assert sl.main(["make-split", "--data", str(root["data"]), "--runs", str(root["data"] / "runs"), "--types", T,
                    "--seeds", "42", "--rawlabels", str(root["raw"])]) == 2
    assert not (root["data"] / "runs").exists()


def test_stripping_the_extra_columns_reproduces_the_production_file_byte_for_byte(root):
    make(root)
    leaky = read(sl.leaky_split_path(root["runs"], T, 42))
    assert list(leaky.columns[-4:]) == ["split_mode", "split_grouped", "split_group", "split"]
    assert set(leaky.split_mode) == {"label_level"}
    assert sl.csv_bytes(sl.strip_leaky(leaky)) == root["split"].read_bytes()


def test_counts_heldout_and_singletons_are_kept_and_panoramas_now_span_splits(root):
    make(root)
    prod, leaky = read(root["split"]), read(sl.leaky_split_path(root["runs"], T, 42))
    assert prod.groupby(["label_class", "split"]).size().equals(leaky.groupby(["label_class", "split"]).size())
    held = prod.split == "test_heldout"
    assert held.any() and (leaky.split[held] == "test_heldout").all() and (leaky.split[~held] != "test_heldout").all()
    size = leaky.groupby("split_group").label_uid.transform("size")
    assert (leaky.split[size == 1] == prod.split[size == 1]).all()
    multi = (size > 1) & ~held
    assert (leaky.split[multi] != prod.split[multi]).any()
    spans = leaky.groupby("pano_id").split.agg(lambda s: {"train", "test"} <= set(s))
    assert spans.any()


def test_a_small_deployment_whose_multi_member_rows_all_sat_in_test_still_scatters():
    rows = []
    for g in range(60):
        for m in range(2):
            rows.append({"label_uid": f"big:{g}-{m}", "city": "big", "pano_id": f"p{g}",
                         "label_class": "incorrect" if (g + m) % 3 == 0 else "correct"})
    df = pd.DataFrame(rows)
    df["split"] = ["train", "train", "train", "train", "train", "train", "train", "val", "val", "test",
                   "test", "test"] * 10
    tiny = pd.DataFrame({"label_uid": [f"tiny:{i}" for i in range(4)], "city": "tiny", "pano_id": "t0",
                         "label_class": ["correct", "incorrect", "correct", "incorrect"], "split": "test"})
    df = pd.concat([df, tiny], ignore_index=True)
    moved = 0
    for seed in range(20):
        new = sl.scatter_splits(df, df.pano_id, seed)
        moved += int((new[df.city == "tiny"] != "test").any())
    assert moved >= 15          # permuted across the whole type, never within one deployment


def test_same_seed_same_bytes_other_seed_changes_split_only_and_the_stream_is_its_own(root):
    make(root, 42)
    first = sl.leaky_split_path(root["runs"], T, 42).read_bytes()
    make(root, 42)
    assert sl.leaky_split_path(root["runs"], T, 42).read_bytes() == first
    make(root, 43)
    a, b = read(sl.leaky_split_path(root["runs"], T, 42)), read(sl.leaky_split_path(root["runs"], T, 43))
    assert a.split_grouped.equals(b.split_grouped) and a.split_group.equals(b.split_group)
    assert not a.split.equals(b.split)
    assert not np.array_equal(sl.scatter_rng(42).integers(0, 2 ** 31, 8),
                              np.random.default_rng(42).integers(0, 2 ** 31, 8))
    prod = read(root["split"])
    own = sl.scatter_splits(prod, a.split_group, 42)
    naive = sl.scatter_splits(prod, a.split_group, 42, rng=np.random.default_rng(42))
    assert (own == a.split.to_numpy()).all() and not (own == naive).all()


# ---------------------------------------------------------------------------------------------------------------------
# Exposure, census and check
# ---------------------------------------------------------------------------------------------------------------------

def _north(m):
    return 47.0 + m / M_PER_DEG_LAT


def test_exposure_on_a_hand_made_frame():
    # A-B-C a chain 2.5 m apart on three panoramas (A to C is 5 m: not direct); D and E share a panorama;
    # F alone; G and H share a panorama, both outside train.
    df = pd.DataFrame({
        "label_uid": list("ABCDEFGH"),
        "pano_id": ["pa", "pb", "pc", "pd", "pd", "pf", "pg", "pg"],
        "label_class": ["correct", "correct", "incorrect", "incorrect", "correct", "correct", "correct", "correct"],
        "split": ["test", "val", "train", "test", "train", "test", "test", "val"],
        "split_group": ["pa", "pa", "pa", "pd", "pd", "pf", "pg", "pg"]})
    coords = pd.DataFrame({"latitude": [_north(0), _north(2.5), _north(5.0), _north(100), _north(100.5),
                                        _north(300), _north(500), _north(500.4)],
                           "longitude": [-122.0] * 8}, index=df.label_uid)
    e = sl.exposure_table(df, coords).set_index("label_uid")
    assert e.loc["A", "k"] == 1 and e.loc["A", "exposure"] == "chain_only"        # its only train link is C, 5 m
    assert e.loc["B", "k"] == 2 and e.loc["B", "exposure"] == "exposed"
    assert e.loc["B", "via"] == "nearby" and e.loc["B", "companion"] == "opposite"
    assert e.loc["D", "k"] == 1 and e.loc["D", "exposure"] == "exposed"
    assert e.loc["D", "via"] == "pano" and e.loc["D", "companion"] == "opposite"
    assert e.loc["F", "k"] == 0 and e.loc["F", "exposure"] == "singleton"
    assert e.loc["G", "k"] == 1 and e.loc["G", "exposure"] == "unexposed"
    assert e.loc["C", "exposure"] == "train"
    no_xy = sl.exposure_table(df, None).set_index("label_uid")                       # panorama companions only
    assert no_xy.loc["B", "k"] == 0 and no_xy.loc["D", "k"] == 1


def test_blank_pano_ids_are_not_one_panorama_for_exposure():
    """Rows with no ``pano_id`` are companions only by distance, never of each other through the blank id."""
    df = pd.DataFrame({"label_uid": list("ABCD"), "pano_id": ["", "", "", "pz"],
                       "label_class": ["correct"] * 4, "split": ["test", "train", "val", "train"],
                       "split_group": list("ABCD")})
    i, j, _ = sl.direct_pairs(df)
    assert len(i) == 0
    e = sl.exposure_table(df, None).set_index("label_uid")
    assert e.loc["A", "k"] == 0 and e.loc["A", "exposure"] == "singleton"
    coords = pd.DataFrame({"latitude": [_north(0), _north(1.0), _north(400), _north(800)],
                           "longitude": [-122.0] * 4}, index=df.label_uid)
    e = sl.exposure_table(df, coords).set_index("label_uid")
    assert e.loc["A", "k"] == 1 and e.loc["A", "exposure"] == "exposed" and e.loc["A", "via"] == "nearby"
    assert e.loc["C", "k"] == 0


def test_census_against_the_iid_expectation():
    e = pd.DataFrame({"split": ["test"] * 4 + ["val", "train"], "k": [1, 1, 2, 0, 3, 1],
                      "exposure": ["exposed", "unexposed", "exposed", "singleton", "exposed", "train"]})
    c = sl.census(e)
    expected = 0.7 + 0.7 + (1 - 0.09) + (1 - 0.027)
    assert c["observed"] == 3 and c["expected"] == pytest.approx(expected)
    assert c["fraction"] == pytest.approx(3 / expected) and c["ok"] is True
    c = sl.census(e.assign(exposure=["unexposed"] * 3 + ["singleton", "unexposed", "train"]))
    assert c["ok"] is False


def test_check_writes_exposure_and_design_and_stops_on_a_coupling_mismatch(root):
    for seed in (42, 43):
        make(root, seed)
    design = sl.check(root["data"], root["runs"], [T], [42, 43], root["raw"], root["out"])
    assert os.path.exists(sl.exposure_path(root["runs"], T, 42))
    on_disk = json.loads((root["out"] / "design.json").read_text())
    assert on_disk == json.loads(json.dumps(design))
    t = on_disk["types"][T]
    assert t["production_split_sha256"] == hashlib.sha256(root["split"].read_bytes()).hexdigest()
    assert set(t["leaky_split_sha256"]) == {"42", "43"} and t["census"]["42"]["ok"] is True
    assert on_disk["heldout_deployments"] == {"count": 1}
    assert on_disk["cross_type_test_share"] == {T: 0.0}         # one type: nothing links it to another
    # a coupling mismatch: one grouped label changed
    path = sl.leaky_split_path(root["runs"], T, 43)
    leaky = read(path)
    j = leaky.index[leaky.split_grouped == "val"][0]
    leaky.loc[j, "split_grouped"] = "test"
    leaky.to_csv(path, index=False, lineterminator="\n")
    os.remove(root["out"] / "design.json")
    with pytest.raises(sl.SplitLeakError, match="coupling"):
        sl.check(root["data"], root["runs"], [T], [42, 43], root["raw"], root["out"])
    assert not (root["out"] / "design.json").exists()


def test_check_sizes_the_link_between_types(root):
    """A second type on the same panoramas: half its test panoramas also carry a test row of the first."""
    import shutil
    make(root, 42)
    other = "Toy2"
    shutil.copytree(root["data"] / T, root["data"] / other)
    os.makedirs(sl.leaky_split_path(root["runs"], other, 42).parent)
    shutil.copy(sl.leaky_split_path(root["runs"], T, 42), sl.leaky_split_path(root["runs"], other, 42))
    design = sl.check(root["data"], root["runs"], [T, other], [42], root["raw"], root["out"])
    assert design["cross_type_test_share"] == {T: 1.0, other: 1.0}
    # rename half of the second type's test panoramas (in both files, so the coupling holds): they link no more
    split = read(root["split"])
    test = split[split.split == "test"]
    renamed = set(test.pano_id.unique()[::2])
    for path in (root["data"] / other / "split.csv", sl.leaky_split_path(root["runs"], other, 42)):
        df = read(path)
        df["pano_id"] = [p + "x" if p in renamed else p for p in df.pano_id]
        df.to_csv(path, index=False, lineterminator="\n")
    design = sl.check(root["data"], root["runs"], [T, other], [42], root["raw"], root["out"])
    expected = float((~test.pano_id.isin(renamed)).mean())
    assert 0 < expected < 1
    assert design["cross_type_test_share"] == pytest.approx({T: expected, other: expected})


def test_check_stops_on_a_census_under_the_threshold(root):
    make(root)
    path = sl.leaky_split_path(root["runs"], T, 42)
    leaky = read(path)
    leaky["split"] = leaky["split_grouped"]        # marked, coupled, but nothing scattered
    leaky.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(sl.SplitLeakError, match="census"):
        sl.check(root["data"], root["runs"], [T], [42], root["raw"], root["out"])
    assert sl.main(["check", "--data", str(root["data"]), "--runs", str(root["runs"]), "--types", T, "--seeds",
                    "42", "--rawlabels", str(root["raw"]), "--out", str(root["out"])]) == 2


# ---------------------------------------------------------------------------------------------------------------------
# Pooling, AUC, bound
# ---------------------------------------------------------------------------------------------------------------------

def test_cmh_pooling_by_hand_and_a_thin_stratum_is_blank():
    strata = pd.DataFrame({"n_exp": [100, 50, 80], "n_unexp": [60, 40, 12],
                           "err_exp": [0.10, 0.20, 0.0], "err_unexp": [0.20, 0.25, 0.5]})
    pooled, blank = sl.cmh_pool(strata)
    w1, w2 = 100 * 60 / 160, 50 * 40 / 90
    assert pooled == pytest.approx((w1 * -0.10 + w2 * -0.05) / (w1 + w2))
    assert blank.tolist() == [False, False, True]           # 12 unexposed < MIN_UNEXPOSED
    pooled_all, blank_all = sl.cmh_pool(strata, min_unexposed=10)
    w3 = 80 * 12 / 92
    assert pooled_all == pytest.approx((w1 * -0.10 + w2 * -0.05 + w3 * -0.5) / (w1 + w2 + w3))
    assert not blank_all.any()


def test_weighted_auc_matches_sklearn_and_replication():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(60) < 0.4
    s = np.round(rng.random(60), 1)                          # ties on purpose
    w = rng.integers(0, 4, 60)
    out = sl.weighted_auc(np.vstack([np.ones(60), w]), y, s)
    assert out[0] == pytest.approx(roc_auc_score(y, s))
    rep = np.repeat(np.arange(60), w)
    assert out[1] == pytest.approx(roc_auc_score(y[rep], s[rep]))


def test_memorisation_bound_on_a_hand_made_case():
    # two singletons and two multi-member rows whose one companion is "correct"; every companion in train
    y = np.array([False, True, False, True])
    score = np.array([0.6, 0.4, 0.9, 0.1])                  # the grouped model ranks every pair wrongly: AUC 0
    comps = [np.array([], bool), np.array([], bool), np.array([False]), np.array([False])]
    b = sl.memorisation_bound(y, score, comps, p_train=1.0, n_draws=3, rng=np.random.default_rng(0))
    assert b["auc_grouped"] == 0.0
    assert b["perfect"] == pytest.approx(0.75)             # m1 -> 0, m2 -> 1: three of four pairs right
    assert b["companion"] == pytest.approx(0.375)          # m2 memorises its companion's class: a tie with m1


def test_planning_figures_by_hand():
    one = pd.DataFrame({"type": ["A"], "k": [1], "n_exp": [100], "n_unexp": [100]})
    mdd = sl.min_detectable_difference(one, {"A": 0.2}, n_seeds=1)
    assert mdd == pytest.approx((1.959964 + 0.841621) * np.sqrt(0.16 / 50), rel=1e-5)
    assert sl.min_detectable_difference(one, {"A": 0.2}, n_seeds=3) == pytest.approx(mdd / np.sqrt(3), rel=1e-9)
    thin = pd.DataFrame({"type": ["A"], "k": [1], "n_exp": [100], "n_unexp": [10]})
    assert np.isnan(sl.min_detectable_difference(thin, {"A": 0.2}))
    hw = sl.expected_halfwidth(sd_seed=0.006, sd_boot=0.009, multi_share=0.3, n_seeds=3)
    assert hw == pytest.approx(1.959964 * np.sqrt(2 * 0.006 ** 2 / 3 + 2 * 0.3 * 0.009 ** 2), rel=1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic studies: null coverage, a memorising model, a model hurt by exposure
# ---------------------------------------------------------------------------------------------------------------------

def synthetic_type(rng, name="A", n_groups=700, leak=0.0, seeds=(42, 43, 44), g_seeds=(42, 43, 44), noise=0.2):
    """One label type: groups of 1-5 labels on one panorama, a grouped split by whole group, scattered L splits.
    Predictions: logit = 1.2 * (2y - 1) + a per-row difficulty shared by every model + small per-model noise; an L
    model adds ``leak * (2y - 1)`` on its exposed rows (memorisation; negative = exposure hurts)."""
    sizes = rng.choice([1, 2, 3, 4, 5], p=[0.4, 0.25, 0.15, 0.12, 0.08], size=n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    y = rng.random(n) < 0.45
    gsplit = rng.choice(["train", "val", "test", "test_heldout"], p=[0.63, 0.135, 0.135, 0.1], size=n_groups)
    rows = pd.DataFrame({"label_uid": [f"{CITIES[g % 3]}:{i}" for i, g in enumerate(group)],
                         "pano_id": [f"p{g}" for g in group], "label_class": np.where(y, "incorrect", "correct"),
                         "split": gsplit[group], "group": [f"p{g}" for g in group]})
    difficulty = rng.normal(0, 1, n)

    def model(extra=0.0):
        logit = 1.2 * (2 * y - 1) + difficulty + rng.normal(0, noise, n) + extra
        return 1 / (1 + np.exp(-logit))

    td = sl.TypeData(name, rows)
    for s in seeds:
        split = sl.scatter_splits(rows, rows.group, s)
        expo = sl.exposure_table(rows.assign(split=split, split_group=rows.group), None)
        exposed = (expo.exposure == "exposed").to_numpy()
        td.add_l(s, split, expo, model(np.where(exposed, leak * (2 * y - 1), 0.0)))
    for s in g_seeds:
        td.add_g(s, model())
    return td


def test_null_simulation_covers_zero_for_the_contrast_and_the_inflation():
    """Equal-quality arms and no memorisation: the deciding 95% interval covers 0 in the plan's band (0.88-0.99) over
    200 seeded replications, for the contrast (groups only, seeds fixed) and the inflation (groups and seeds).
    ``n_boot`` is 400: at 200 the percentile endpoints under-cover (review round 1 measured 0.93)."""
    rng = np.random.default_rng(7)
    hits_c = hits_i = 0
    reps = 200
    for r in range(reps):
        td = synthetic_type(rng, n_groups=1200, noise=0.5)
        est = sl.estimate([td], n_boot=400, boot_seed=r, min_unexposed=10)
        c, i = est["contrast"]["error"], est["inflation"]["test"]["pooled"]
        hits_c += c["lo"] <= 0 <= c["hi"]
        hits_i += i["lo"] <= 0 <= i["hi"]
    assert 0.88 <= hits_c / reps <= 0.99, hits_c
    assert 0.88 <= hits_i / reps <= 0.99, hits_i


def test_only_the_across_arm_deltas_resample_seeds(monkeypatch):
    """The contrasts resample groups with the seeds held fixed; the inflation and the other L - G deltas also
    resample seeds. Pinning every seed draw to the identity leaves the contrasts' intervals unchanged and changes
    the deltas'."""
    td = synthetic_type(np.random.default_rng(5), n_groups=1500)
    a = sl.estimate([td], n_boot=200, boot_seed=1, min_unexposed=10)
    monkeypatch.setattr(sl, "_seed_draws", lambda n, b, rng: np.tile(np.arange(n), (b + 1, 1)))
    b = sl.estimate([td], n_boot=200, boot_seed=1, min_unexposed=10)
    for metric in ("error", "p_true"):
        assert (a["contrast"][metric]["lo"], a["contrast"][metric]["hi"]) == \
            (b["contrast"][metric]["lo"], b["contrast"][metric]["hi"])
        assert a["contrast_descriptive"]["via_pano"][metric]["lo"] == b["contrast_descriptive"]["via_pano"][metric]["lo"]
    for key in ("inflation", "heldout_delta", "singleton_delta"):
        d_a = a[key]["test"]["pooled"] if key == "inflation" else a[key]["pooled"]
        d_b = b[key]["test"]["pooled"] if key == "inflation" else b[key]["pooled"]
        assert d_a["point"] == d_b["point"] and (d_a["lo"], d_a["hi"]) != (d_b["lo"], d_b["hi"]), key


def test_a_contrast_from_one_estimable_seed_is_provisional():
    """Review round 1, B2: two of three seeds have every stratum blank, so the t interval does not exist. The reading
    must say provisional, not give a confirmatory label from one seed's data."""
    rng = np.random.default_rng(7)
    for _ in range(121):
        td = synthetic_type(rng, n_groups=1200, noise=0.5)
    est = sl.estimate([td], n_boot=20, boot_seed=120, min_unexposed=10)
    c = est["contrast"]["error"]
    assert sum(v is not None for v in c["per_seed"].values()) == 1 and c["t_lo"] is None
    rule = sl.rule_outcome(c, est["inflation"]["test"]["pooled"], 3, "fresh")
    assert rule["contrast"].endswith("(provisional: 1 of 3 seeds)"), rule["contrast"]
    assert rule["provisional"] is True and rule["provisional_contrast"] is True
    assert rule["estimable_seeds"]["contrast"] == 1


@pytest.mark.parametrize("leak, outcome", [(3.0, "leak shown"), (-3.0, "exposure hurts")])
def test_a_memorising_model_shows_a_leak_and_a_hurt_model_shows_harm(leak, outcome):
    rng = np.random.default_rng(1)
    types = [synthetic_type(rng, name, leak=leak) for name in ("A", "B")]
    est = sl.estimate(types, n_boot=300, boot_seed=0, min_unexposed=10)
    rule = sl.rule_outcome(est["contrast"]["error"], est["inflation"]["test"]["pooled"], n_seeds=3,
                           g_source="fresh")
    assert rule["contrast"] == outcome and rule["provisional"] is False
    assert est["contrast"]["p_true"]["point"] * np.sign(leak) > 0     # the second metric agrees in sign
    assert set(est["contrast_by_type"]) == {"A", "B"}
    used = est["strata"][~est["strata"].blank]
    assert len(used) and set(used.k) <= {1, 2, 3}


def test_estimate_point_equals_cmh_over_its_own_strata():
    td = synthetic_type(np.random.default_rng(3), n_groups=1500, seeds=(42,), g_seeds=(42,))
    est = sl.estimate([td], n_boot=20, boot_seed=0, min_unexposed=10)
    st = est["strata"]
    pooled, blank = sl.cmh_pool(st, min_unexposed=10)
    assert (~blank).any() and blank.equals(st.blank)
    assert est["contrast"]["error"]["point"] == pytest.approx(pooled)
    assert est["contrast"]["error"]["t_lo"] is None           # one seed: no t interval


# ---------------------------------------------------------------------------------------------------------------------
# The pre-registered rule
# ---------------------------------------------------------------------------------------------------------------------

def ci(lo, hi, t_lo=None, t_hi=None, **extra):
    return {"point": (lo + hi) / 2, "lo": lo, "hi": hi, "t_lo": t_lo, "t_hi": t_hi, **extra}


LEAK = ci(-0.05, -0.01, -0.06, -0.005)          # a confirmatory contrast (both intervals below 0)
FLAT = ci(-0.001, 0.002, -0.002, 0.003)         # a confirmatory "no material inflation"


@pytest.mark.parametrize("contrast, label", [
    (ci(-0.05, -0.01, -0.06, -0.001), "leak shown"),
    (ci(0.01, 0.04, 0.001, 0.05), "exposure hurts"),
    (ci(-0.02, 0.01, -0.03, 0.02), "not shown (95% interval [-0.0200, +0.0100])"),
    (ci(-0.05, -0.01, -0.07, 0.01), "unresolved (the bootstrap and t intervals disagree)"),
    (ci(-0.05, -0.01), "leak shown (provisional: no t interval)"),
])
def test_rule_contrast_branches(contrast, label):
    assert sl.rule_outcome(contrast, FLAT, n_seeds=3, g_source="fresh")["contrast"] == label


@pytest.mark.parametrize("inflation, label", [
    (ci(0.002, 0.02, 0.001, 0.03), "shown"),
    (ci(-0.004, 0.005, -0.005, 0.0055), "no material inflation"),
    (ci(-0.004, 0.009, -0.005, 0.01), "unresolved, upper bound +0.0090"),
    (ci(-0.004, 0.005, -0.008, 0.009), "unresolved, upper bound +0.0090 (the bootstrap and t intervals disagree)"),
    (ci(-0.02, 0.005, -0.03, 0.005), "unresolved, upper bound +0.0050"),      # the lower side of the band
    (ci(-0.004, 0.005), "no material inflation (provisional: no t interval)"),
])
def test_rule_inflation_branches(inflation, label):
    assert sl.rule_outcome(LEAK, inflation, n_seeds=3, g_source="fresh")["inflation"] == label


def test_rule_labels_for_stored_baselines_and_one_seed():
    shown = ci(0.002, 0.02, 0.001, 0.03)
    r = sl.rule_outcome(LEAK, shown, n_seeds=3, g_source="stored")
    assert r["inflation"] == "shown (descriptive: runs from different dates and environments)"
    assert r["contrast"] == "leak shown" and r["provisional"] is False and r["confirmatory_inflation"] is False
    r = sl.rule_outcome(ci(-0.05, -0.01), ci(0.002, 0.02), n_seeds=1, g_source="fresh")
    assert r["contrast"] == "leak shown (provisional: 1 of 3 seeds)"
    assert r["inflation"] == "shown (provisional: 1 of 3 seeds)" and r["provisional"] is True
    with pytest.raises(ValueError):
        sl.rule_outcome(LEAK, shown, n_seeds=3, g_source="old")


def test_rule_counts_estimable_seeds_from_the_data():
    """Seeds are counted from ``per_seed`` (contrast) and the smaller arm (inflation), never from the plan alone;
    a t interval at two seeds still decides disagreement, and provisional is added to it, not put in its place."""
    two = {"42": -0.02, "43": None, "44": -0.03}
    r = sl.rule_outcome(ci(-0.05, -0.01, -0.07, 0.01, per_seed=two), FLAT, n_seeds=3, g_source="fresh")
    assert r["contrast"] == "unresolved (the bootstrap and t intervals disagree) (provisional: 2 of 3 seeds)"
    assert r["provisional_contrast"] is True and r["provisional_inflation"] is False
    r = sl.rule_outcome(ci(-0.05, -0.01, -0.06, -0.005, per_seed=two), FLAT, n_seeds=3, g_source="fresh")
    assert r["contrast"] == "leak shown (provisional: 2 of 3 seeds)"
    full = {"42": 0.9, "43": 0.91, "44": 0.92}
    infl = ci(0.002, 0.02, 0.001, 0.03, per_seed_l=full, per_seed_g={"42": 0.9, "43": None, "44": None})
    r = sl.rule_outcome(LEAK, infl, n_seeds=3, g_source="fresh")
    assert r["inflation"] == "shown (provisional: 1 of 3 seeds)" and r["confirmatory_inflation"] is False
    assert r["estimable_seeds"] == {"contrast": 3, "inflation": 1}
    r = sl.rule_outcome(ci(-0.05, -0.01, -0.06, -0.005, per_seed=dict.fromkeys(full)), FLAT, 3, "fresh")
    assert r["provisional"] is True                                     # a contrast with no estimable seed
    blank = {"point": None, "lo": None, "hi": None, "t_lo": None, "t_hi": None, "per_seed": dict.fromkeys(full)}
    r = sl.rule_outcome(blank, FLAT, n_seeds=3, g_source="fresh")
    assert r["contrast"].startswith("not estimable") and "provisional" not in r["contrast"]
    assert r["provisional_contrast"] is True and r["provisional_inflation"] is False
    r = sl.rule_outcome(LEAK, FLAT, n_seeds=3, g_source="fresh")
    assert r["provisional"] is False and r["confirmatory_inflation"] is True


# ---------------------------------------------------------------------------------------------------------------------
# The whole CPU path on the toy root: bound and report, thin-n blanks, nothing that names a deployment
# ---------------------------------------------------------------------------------------------------------------------

def fake_predictions(split_df, rng, path, exposed=()):
    ev = split_df[split_df.split.isin(["val", "test", "test_heldout"])]
    y = (ev.label_class == "incorrect").to_numpy()
    p = 1 / (1 + np.exp(-(1.5 * (2 * y - 1) + rng.normal(0, 1, len(ev)))))
    p = np.where(ev.label_uid.isin(set(exposed)), np.where(y, 0.99, 0.01), p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class, "p_correct": 1 - p,
                  "p_incorrect": p}).to_csv(path, index=False)


def test_bound_and_report_on_the_toy_root_blank_thin_strata_and_name_no_deployment(root):
    rng = np.random.default_rng(0)
    make(root, 42)
    sl.check(root["data"], root["runs"], [T], [42], root["raw"], root["out"])
    prod = read(root["split"])
    fake_predictions(prod, rng, root["data"] / T / "predictions.csv")
    leaky = read(sl.leaky_split_path(root["runs"], T, 42))
    expo = read(sl.exposure_path(root["runs"], T, 42))
    fake_predictions(leaky, rng, sl.run_dir(root["runs"], T, "L", 42) / "predictions.csv",
                     exposed=expo.label_uid[expo.exposure == "exposed"])
    assert sl.main(["bound", "--data", str(root["data"]), "--runs", str(root["runs"]), "--types", T,
                    "--rawlabels", str(root["raw"]), "--out", str(root["out"]), "--n-boot", "50"]) == 0
    b = json.loads((root["out"] / "bound.json").read_text())
    assert set(b["types"][T]) >= {"memorisation_bound", "companion_bound", "expected_halfwidth",
                                  "min_detectable_difference"}
    assert sl.main(["report", "--data", str(root["data"]), "--runs", str(root["runs"]), "--types", T, "--seeds",
                    "42", "--g-seeds", "42", "--g-source", "stored", "--out", str(root["out"]),
                    "--n-boot", "50"]) == 0
    res = json.loads((root["out"] / "results.json").read_text())
    assert res["rule"]["contrast"].startswith("not estimable") and res["rule"]["provisional"] is True
    assert res["rule"]["inflation"].endswith("(provisional: 1 of 3 seeds)")
    assert res["rule"]["estimable_seeds"] == {"contrast": 0, "inflation": 1}
    assert any(s["blank"] for s in res["strata"])            # the toy has fewer than 30 unexposed rows per stratum
    tables = (root["out"] / "tables.md").read_text()
    assert "blank" in tables and "fewer than 30 unexposed" in tables
    assert (root["out"] / "slices.md").exists() and (root["out"] / "seeds.md").exists()
    names = re.compile("|".join(map(re.escape, CITIES)), re.IGNORECASE)
    written = sorted(p for p in root["out"].iterdir() if p.is_file())
    assert {p.name for p in written} >= {"design.json", "bound.json", "bound.md", "tables.md", "seeds.md",
                                         "slices.md", "results.json"}
    for p in written:
        assert not names.search(p.read_text()), p.name


def test_report_refuses_predictions_that_do_not_cover_the_arm(root):
    rng = np.random.default_rng(0)
    make(root, 42)
    sl.check(root["data"], root["runs"], [T], [42], root["raw"], root["out"])
    prod = read(root["split"])
    fake_predictions(prod, rng, root["data"] / T / "predictions.csv")
    fake_predictions(prod, rng, sl.run_dir(root["runs"], T, "L", 42) / "predictions.csv")   # G's rows, not L's
    with pytest.raises(sl.SplitLeakError, match="cover"):
        sl.load_study(root["data"], root["runs"], [T], [42], [42], "stored")


def test_predictions_with_a_duplicated_row_are_refused(root):
    prod = read(root["split"])
    path = root["data"] / T / "predictions.csv"
    fake_predictions(prod, np.random.default_rng(0), path)
    pred = pd.read_csv(path, dtype=str)
    pd.concat([pred, pred.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(sl.SplitLeakError, match="cover exactly"):
        sl._read_predictions(path, prod, prod["split"], "G")


def test_bound_stops_before_make_split_and_check(root):
    fake_predictions(read(root["split"]), np.random.default_rng(0), root["data"] / T / "predictions.csv")
    args = ["bound", "--data", str(root["data"]), "--runs", str(root["runs"]), "--types", T,
            "--rawlabels", str(root["raw"]), "--out", str(root["out"]), "--n-boot", "10"]
    with pytest.raises(sl.SplitLeakError, match="run make-split first"):
        sl.bound(root["data"], root["runs"], [T], root["raw"], root["out"], n_boot=10)
    make(root, 42)
    with pytest.raises(sl.SplitLeakError, match="run check first"):
        sl.bound(root["data"], root["runs"], [T], root["raw"], root["out"], n_boot=10)
    assert sl.main(args) == 2 and not (root["out"] / "bound.json").exists()


def test_make_split_refuses_runs_under_the_data_directory(root, capsys):
    assert sl.main(["make-split", "--data", str(root["data"]), "--runs", str(root["data"] / "runs"), "--types", T,
                    "--seeds", "42", "--rawlabels", str(root["raw"])]) == 2
    assert "--runs" in capsys.readouterr().err          # main's own guard, before make_split's per-file one
    assert not (root["data"] / "runs").exists()


# ---------------------------------------------------------------------------------------------------------------------
# The doc quotes the code's constants
# ---------------------------------------------------------------------------------------------------------------------

def test_the_doc_quotes_the_constants_and_the_rule():
    text = open(DOC).read()
    assert f"±{sl.INFLATION_MARGIN}" in text and f"{sl.N_BOOT:,}" in text
    assert f"bootstrap seed {sl.BOOT_SEED}" in text
    assert f"fewer than {sl.MIN_UNEXPOSED} unexposed" in text
    assert f"{sl.CENSUS_MIN_FRACTION:.0%}" in text and f"precision {sl.PRECISION_TARGET}" in text
    assert f"{sl.MERGE_RADIUS_M:g} m" in text and "k = 1, 2, 3+" in text
    assert "three seeds" in text and sl.CONFIRMATORY_SEEDS == 3
    assert "run held, no result exists" in text.lower() or "Run held, no result exists" in text
    for label in ("leak shown", "exposure hurts", "not shown", "no material inflation", "unresolved",
                  "descriptive: runs from different dates and environments", "(provisional: 1 of 3 seeds)",
                  "(provisional: k of 3 seeds)", "(provisional: no t interval)", "Amendment, 2026-11-23"):
        assert label in text, label
