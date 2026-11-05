"""analysis/region_transfer.py: LOCO variants, fine-tune pools, predict wiring, calibration and the report."""
import doctest
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import confident_learning as cl  # noqa: E402
import region_transfer as rt  # noqa: E402
import regions  # noqa: E402

import data_generation  # noqa: E402

# Exactly the round-3 split.csv columns: data_generation.MANIFEST_COLUMNS with split_data.add_hard's ``hard`` before
# ``crop_path`` and ``split`` last. No latitude/longitude: the 3 m merge reads them from the rawLabels cache, as on klone.
_AT = data_generation.MANIFEST_COLUMNS.index("crop_path")
COLS = data_generation.MANIFEST_COLUMNS[:_AT] + ["hard"] + data_generation.MANIFEST_COLUMNS[_AT:] + ["split"]
# xland: alpha, beta; yland: gamma (public), dorado (private); wland: epsilon (round-3 train/val/test) and zeta
# (round-3 test_heldout, so a held-out wland must leave zeta's rows byte-identical); zland (private only): omega,
# held out.
CITIES = ["alpha", "beta", "gamma", "dorado", "epsilon", "omega", "zeta"]
HELDOUT_CITIES = {"omega", "zeta"}
LISTING = {"cities": [
    {"city_id": "alpha", "country_id": "xland", "visibility": "public"},
    {"city_id": "beta", "country_id": "xland", "visibility": "public"},
    {"city_id": "gamma", "country_id": "yland", "visibility": "public"},
    {"city_id": "dorado", "country_id": "yland", "visibility": "private", "city_name_short": "Doradoville"},
    {"city_id": "epsilon", "country_id": "wland", "visibility": "public"},
    {"city_id": "omega", "country_id": "zland", "visibility": "private", "city_name_short": "Omegaville"},
    {"city_id": "zeta", "country_id": "wland", "visibility": "public"},
]}
LT = "Toy"


def make_data(tmp_path, n_panos=490, seed=0):
    """``$DATA`` with ``Toy/split.csv`` (round-3 columns, no coordinates), crops, ``cities.json`` and the
    rawLabels cache ``rawlabels/<city>.csv`` (``label_id``, ``latitude``, ``longitude``)."""
    rng = np.random.default_rng(seed)
    rows, coords, i = [], [], 0
    for p in range(n_panos):
        city = CITIES[p % len(CITIES)]
        split = ("test_heldout" if city in HELDOUT_CITIES
                 else ["train", "train", "train", "val", "test"][(p // len(CITIES)) % 5])
        for _ in range(int(rng.integers(1, 4))):
            cls = "incorrect" if rng.random() < 0.45 else "correct"
            rows.append({"label_uid": f"{city}:{i}", "city": city, "label_id": str(i), "pano_id": f"pano{p:04d}",
                         "label_type": LT, "label_class": cls, "class_source": "margin", "author": "human",
                         "human_agree": "3", "human_disagree": "0", "severity": str(int(rng.integers(1, 4))),
                         "tags": json.dumps(["steep"] if rng.random() < 0.5 else []), "hard": "0",
                         "crop_path": f"{cls}/{city}_{i}.webp", "status": "ok", "split": split})
            coords.append({"city": city, "label_id": str(i), "latitude": f"{47.0 + p * 0.001:.6f}",
                           "longitude": "-122.300000"})
            i += 1
    df = pd.DataFrame(rows, columns=COLS).astype(object).fillna("").sort_values("label_uid").reset_index(drop=True)
    base = tmp_path / LT
    base.mkdir(parents=True, exist_ok=True)
    df.to_csv(base / "split.csv", index=False, lineterminator="\n")
    raw = tmp_path / "rawlabels"
    raw.mkdir(exist_ok=True)
    for city, g in pd.DataFrame(coords).groupby("city"):
        g.drop(columns="city").to_csv(raw / f"{city}.csv", index=False)
    (tmp_path / "cities.json").write_text(json.dumps(LISTING))
    return tmp_path, df


@pytest.fixture
def data(tmp_path):
    return make_data(tmp_path)


def country_series(df):
    return regions.assign(df[["city"]], regions.country_table(LISTING, {}))["country"]


def run_variants(data_dir, *extra):
    return rt.main(["variants", "--split", str(data_dir / LT / "split.csv"), "--cities-json",
                    str(data_dir / "cities.json"), "--rawlabels", str(data_dir / "rawlabels"),
                    "--min-eval-class", "20", "--min-pool", "20",
                    "--min-frac-class", "5", "--fractions", "0.25,0.5,1.0", "--out",
                    str(data_dir / LT / "region"), *extra])


# ---------------------------------------------------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------------------------------------------------

def test_loco_moves_only_the_country_and_keeps_everything_else_byte_identical(data):
    data_dir, df = data
    assert run_variants(data_dir) == 0
    loco = cl.read_split(str(data_dir / LT / "region" / "yland" / "split.csv"))
    base = cl.read_split(str(data_dir / LT / "split.csv"))
    y = base["city"].isin(["gamma", "dorado"])
    m = base.merge(loco[["label_uid", "split"]], on="label_uid", suffixes=("", "_v"), validate="one_to_one")
    assert (m.loc[y & (m.split == "train"), "split_v"] == rt.POOL).all()
    assert (m.loc[y & m.split.isin(["val", "test"]), "split_v"] == rt.HELDOUT_COUNTRY).all()
    assert (m.loc[m.city == "omega", "split_v"] == "test_heldout").all()
    other = base[~y].reset_index(drop=True)
    assert other.equals(loco[loco["label_uid"].isin(other["label_uid"])].reset_index(drop=True)[other.columns])
    assert list(loco.columns) == list(base.columns)
    assert (data_dir / LT / "region" / "yland" / "split.csv").read_bytes().count(b"\r") == 0


def test_fixture_mirrors_the_round3_split_and_variants_needs_rawlabels(data):
    # The round-3 split.csv has no coordinates; without the rawLabels cache the 3 m merge cannot run.
    data_dir, _ = data
    head = (data_dir / LT / "split.csv").read_text().split("\n")[0].split(",")
    assert head == COLS and "latitude" not in head
    with pytest.raises(ValueError, match="no latitude/longitude"):
        rt.main(["variants", "--split", str(data_dir / LT / "split.csv"), "--cities-json",
                 str(data_dir / "cities.json"), "--min-eval-class", "20", "--min-pool", "20", "--min-frac-class",
                 "5", "--out", str(data_dir / LT / "region")])


def test_round3_heldout_rows_of_the_heldout_country_stay_byte_identical(data):
    data_dir, df = data
    assert run_variants(data_dir) == 0
    rdir = data_dir / LT / "region"
    assert json.loads((rdir / "variants.json").read_text())["countries"]["wland"]["eligible"]
    base = cl.read_split(str(data_dir / LT / "split.csv"))
    zeta = base[base.city == "zeta"].reset_index(drop=True)
    assert len(zeta) and (zeta.split == "test_heldout").all()
    files = [rdir / "wland" / "split.csv"] + sorted((rdir / "wland").glob("ft-*/split.csv"))
    assert len(files) >= 2
    for f in files:
        v = cl.read_split(str(f))
        got = v[v.label_uid.isin(zeta.label_uid)].sort_values("label_uid").reset_index(drop=True)
        assert zeta.sort_values("label_uid").reset_index(drop=True).equals(got[zeta.columns]), f
    loco = cl.read_split(str(files[0]))
    assert (loco.loc[loco.city == "epsilon", "split"] != "train").all()   # the country's other rows did move


def test_variants_refuses_a_loco_split_that_touches_round3_heldout_rows(data, monkeypatch):
    # The untouched mask must cover the held-out country's round-3 test_heldout rows, not only other countries.
    data_dir, _ = data
    real = rt.loco_split

    def leaky(base, country_col, country):
        out = real(base, country_col, country)
        out.loc[out.city == "zeta", "split"] = rt.HELDOUT_COUNTRY
        return out
    monkeypatch.setattr(rt, "loco_split", leaky)
    with pytest.raises(AssertionError, match="changed rows"):
        run_variants(data_dir, "--countries", "wland")


def test_check_rows_unchanged_raises_on_a_planted_change():
    base = pd.DataFrame({"label_uid": ["a", "b", "c"], "split": ["train", "val", "test"], "severity": ["1", "2", "3"]})
    keep = np.array([True, True, False])
    rt.check_rows_unchanged(base, base.copy(), keep)
    rt.check_rows_unchanged(base, base.assign(split=["train", "val", "pool"]), keep)   # c is not protected
    with pytest.raises(AssertionError):
        rt.check_rows_unchanged(base, base.assign(severity=["1", "9", "3"]), keep)
    with pytest.raises(AssertionError):
        rt.check_rows_unchanged(base, base.iloc[1:], keep)                             # a protected row dropped


@pytest.mark.parametrize("other", ["train", "val", "test"])
def test_check_no_straddle_catches_a_panorama_in_the_heldout_country_split(other):
    df = pd.DataFrame({"label_uid": ["a", "b", "c"], "pano_id": ["p1", "p1", "p2"],
                       "split": [other, rt.HELDOUT_COUNTRY, "pool"]})
    with pytest.raises(Exception):
        rt.check_no_straddle(df)
    rt.check_no_straddle(df.assign(pano_id=["p1", "p3", "p2"]))
    rt.check_no_straddle(df.assign(split=[rt.POOL, rt.HELDOUT_COUNTRY, "pool"]))   # pool rows are not scored


def test_eligibility_thresholds_and_reasons(data):
    _, df = data
    country = country_series(df)
    ok, reason, counts = rt.eligible(df, country, "yland", 20)
    assert ok and reason == ""
    assert counts["eval"]["correct"] >= 20 and counts["heldout"] == {"correct": 0, "incorrect": 0}
    ok, reason, _ = rt.eligible(df, country, "zland", 20)
    assert not ok and reason == "all rows already test_heldout"
    small = df[~((country == "wland") & (df.split.isin(["val", "test"])) & (df.label_class == "incorrect"))
               | (df.label_uid.isin(df[(country == "wland") & (df.split == "test") & (df.label_class == "incorrect")]
                                    .label_uid.head(5)))]
    ok, reason, _ = rt.eligible(small, country_series(small), "wland", 30)
    assert not ok and reason == "eval incorrect 5 < 30"
    ok, reason, _ = rt.eligible(df, country, "xland", 20, max_train_share=0.2)
    assert not ok and reason.startswith("holds ") and reason.endswith("of the train rows")


def test_pool_fractions_are_nested_grouped_and_seeded(data):
    data_dir, df = data
    country = country_series(df)
    loco = rt.loco_split(df, country, "yland")
    pool = loco[loco.split == rt.POOL]
    groups = cl.train_groups(pool, str(data_dir / "rawlabels"), 3.0)
    a = rt.pool_fractions(pool, groups, [0.25, 0.5, 1.0], 42, 0.2, 5)
    s25, s50, s100 = (set(a[p]["train"]) for p in (25, 50, 100))
    assert s25 < s50 < s100
    val = set(a[100]["val"])
    assert all(set(a[p]["val"]) == val for p in a) and not (val & s100)
    assert len(val) + len(s100) == len(pool)
    pano = dict(zip(pool.label_uid, pool.pano_id))
    assert not ({pano[u] for u in val} & {pano[u] for u in s100})
    assert rt.pool_fractions(pool, groups, [0.25, 0.5, 1.0], 42, 0.2, 5) == a
    assert rt.pool_fractions(pool, groups, [0.25, 0.5, 1.0], 7, 0.2, 5)[25]["train"] != a[25]["train"]
    skipped = rt.pool_fractions(pool, groups, [0.25], 42, 0.2, 10_000)[25]
    assert skipped["skipped"].startswith("train ") and skipped["train_counts"]["correct"] > 0


def test_ft_split_trains_only_local_rows(data):
    data_dir, df = data
    run_variants(data_dir)
    ft = cl.read_split(str(data_dir / LT / "region" / "yland" / "ft-50" / "split.csv"))
    country = country_series(ft)
    assert set(ft.loc[ft.split.isin(["train", "val"]), "city"]) <= {"gamma", "dorado"}
    assert (ft.loc[(country != "yland") & df.set_index("label_uid").loc[ft.label_uid, "split"]
                   .isin(["train", "val"]).to_numpy(), "split"] == rt.UNUSED).all()
    loco = cl.read_split(str(data_dir / LT / "region" / "yland" / "split.csv"))
    keep = loco.split.isin(["test", "test_heldout", rt.HELDOUT_COUNTRY])
    assert loco[keep].reset_index(drop=True).equals(
        ft[ft.label_uid.isin(loco[keep].label_uid)].reset_index(drop=True)[loco.columns])
    assert set(ft.split) <= {"train", "val", "test", "test_heldout", rt.HELDOUT_COUNTRY, rt.POOL, rt.UNUSED}


def test_no_panorama_straddles_splits_in_any_written_file(data):
    data_dir, _ = data
    run_variants(data_dir)
    files = list((data_dir / LT / "region").rglob("split.csv"))
    assert len(files) >= 6
    for f in files:
        rt.check_no_straddle(cl.read_split(str(f)))


def test_variants_json_hashes_and_counts_match_the_files(data, capsys):
    data_dir, df = data
    run_variants(data_dir)
    out = capsys.readouterr().out
    assert "not eligible: all rows already test_heldout" in out
    rdir = data_dir / LT / "region"
    meta = json.loads((rdir / "variants.json").read_text())
    assert meta["source_sha256"] == cl.sha256_of(str(data_dir / LT / "split.csv"))
    y = meta["countries"]["yland"]
    assert y["eligible"] and y["deployments"] == 2 and y["split_sha256"] == cl.sha256_of(str(rdir / "yland" / "split.csv"))
    for pct, fr in y["fractions"].items():
        f = rdir / "yland" / f"ft-{pct}" / "split.csv"
        assert fr["split_sha256"] == cl.sha256_of(str(f))
        v = cl.read_split(str(f))
        assert fr["train"] == rt.class_counts(v[v.split == "train"])
    assert not meta["countries"]["zland"]["eligible"]
    assert rt.main(["list", "--variants", str(rdir / "variants.json")]) == 0
    listed = capsys.readouterr().out.split("\n")
    assert any(ln.startswith("yland 25 50 100") for ln in listed)


# ---------------------------------------------------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------------------------------------------------

def fake_scores(manifest, splits, bump=0.0):
    rows = manifest[manifest.split.isin(splits)]
    p = np.where(rows.label_class == "incorrect", 0.7 + bump, 0.3)
    return pd.DataFrame({"label_uid": rows.label_uid.values, "split": rows.split.values,
                         "true": rows.label_class.values, "p_correct": 1 - p, "p_incorrect": p})


def test_predict_wires_plain_and_conditioned_paths(data, monkeypatch, tmp_path):
    import evaluate
    import region_model
    data_dir, _ = data
    run_variants(data_dir)
    calls = []

    def plain(manifest, ckpt, revision=None, splits=evaluate.EVAL_SPLITS):
        calls.append(("plain", ckpt, tuple(splits)))
        return fake_scores(manifest, splits)

    def cond(manifest, ckpt, splits, table, force_unk=False):
        calls.append(("cond", ckpt, tuple(splits), force_unk, table["gamma"]))
        return fake_scores(manifest, splits)
    monkeypatch.setattr(evaluate, "run_inference", plain)
    monkeypatch.setattr(region_model, "run_inference", cond)
    split_file = str(data_dir / LT / "region" / "yland" / "split.csv")
    out = tmp_path / "pred.csv"
    common = ["predict", "--split-file", split_file, "--data-dir", str(data_dir), "--label-type", LT,
              "--checkpoint", "ckpt-x", "--out", str(out)]
    assert rt.main(common) == 0
    assert calls[-1] == ("plain", "ckpt-x", rt.SCORED_SPLITS)
    pred = pd.read_csv(out)
    assert set(pred.split) == {"val", "test", "test_heldout", rt.HELDOUT_COUNTRY, rt.POOL}
    assert pred.label_uid.is_monotonic_increasing
    assert rt.main(common) == 0 and len(calls) == 1   # complete: the requeue guard skips
    out.unlink()
    assert rt.main(common + ["--conditioned", "--cities-json", str(data_dir / "cities.json"), "--force-unk",
                             "--splits", "test,test_heldout_country"]) == 0
    assert calls[-1] == ("cond", "ckpt-x", ("test", rt.HELDOUT_COUNTRY), True, "yland")
    with pytest.raises(SystemExit, match="--cities-json"):
        rt.main(common + ["--conditioned"])
    with pytest.raises(SystemExit, match="--run-dir or --checkpoint"):
        rt.main(["predict", "--split-file", split_file, "--data-dir", "x", "--label-type", LT, "--out", "y"])


# ---------------------------------------------------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------------------------------------------------

def scored(df, rng, signal=1.5):
    """Synthetic predictions for every row of ``df`` (with its split): logit = signal * y + noise."""
    y = (df.label_class == "incorrect").to_numpy()
    p = 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(0, 1, len(df)))))
    return pd.DataFrame({"label_uid": df.label_uid.values, "split": df.split.values, "true": df.label_class.values,
                         "p_correct": 1 - p, "p_incorrect": p})


def test_calibrate_local_map_fits_on_fraction_rows_and_reports_wilson(data, monkeypatch):
    from calibration import calibrate as cal
    from calibration.per_city import wilson_interval
    data_dir, df = data
    run_variants(data_dir)
    rng = np.random.default_rng(0)
    loco = cl.read_split(str(data_dir / LT / "region" / "yland" / "split.csv"))
    pred = scored(loco, rng, 3.0)
    ft50 = cl.read_split(str(data_dir / LT / "region" / "yland" / "ft-50" / "split.csv"))
    uids = {50: ft50.loc[ft50.split == "train", "label_uid"].tolist()}
    seen = []
    real = cal.fit_curves

    def spy(fit_df, pi):
        seen.append(set(fit_df.label_uid))
        return real(fit_df, pi)
    monkeypatch.setattr(cal, "fit_curves", spy)
    pi = {"correct": 0.8, "incorrect": 0.2}
    recs = rt.loco_calibration(pred, uids, pi, min_gated=5)
    assert seen[0] == set(pred[pred.split == "val"].label_uid)
    assert seen[1] == set(uids[50])
    local = [r for r in recs if r["map"] == "local-50"]
    assert {r["verdict"] for r in local} == {"correct", "incorrect"} and local[0]["fit_n"] == len(uids[50])
    ev = cal.verdicts(pred[pred.split == rt.HELDOUT_COUNTRY])
    r = next(x for x in local if x["verdict"] == "correct")
    assert r["n"] == int((ev.verdict == "correct").sum())
    if r["precision_raw"] is not None:
        lo, hi = wilson_interval(r["precision_raw"], r["gated_n"])
        assert r["wilson_lo"] == pytest.approx(float(lo)) and r["wilson_hi"] == pytest.approx(float(hi))
        assert r["coverage"] == pytest.approx(r["gated_n"] / r["n"])
    one_class = pred[pred.true == "correct"]
    assert rt.gate_records(one_class, pred, pi) is None


def test_calibrate_command_writes_loco_and_in_distribution_records(data):
    data_dir, df = data
    run_variants(data_dir)
    rng = np.random.default_rng(1)
    loco = cl.read_split(str(data_dir / LT / "region" / "yland" / "split.csv"))
    path = rt.pred_path(str(data_dir), LT, rt.tag_global("yland", 42))
    os.makedirs(os.path.dirname(path))
    scored(loco, rng).to_csv(path, index=False)
    scored(df, rng).to_csv(data_dir / LT / "predictions.csv", index=False)
    (data_dir / "priors.json").write_text(json.dumps({LT: {"correct": 900, "incorrect": 100}}))
    assert rt.main(["calibrate", "--data-dir", str(data_dir), "--label-type", LT, "--priors",
                    str(data_dir / "priors.json"), "--cities-json", str(data_dir / "cities.json")]) == 0
    out = json.loads((data_dir / LT / "region" / "calibration.json").read_text())
    kinds = {(r["kind"], r["map"]) for r in out["records"]}
    assert ("loco", "pooled") in kinds and ("loco", "local-100") in kinds
    assert ("in_dist", "own") in kinds and ("in_dist", "pooled") in kinds
    assert {r["country"] for r in out["records"] if r["kind"] == "in_dist"} <= {"xland", "yland", "wland"}


# ---------------------------------------------------------------------------------------------------------------------
# report: pure functions
# ---------------------------------------------------------------------------------------------------------------------

def joined(df, pred):
    return pred.merge(df[cl.PRED_JOIN_COLUMNS], on="label_uid", validate="one_to_one")


def test_paired_delta_flags_a_known_difference(data):
    _, df = data
    rng = np.random.default_rng(3)
    test = df[df.split.isin(["test", "val"])].assign(split="test")
    base = joined(test, scored(test, rng, 0.8))
    better = base.copy()
    better.loc[better.true == "incorrect", "p_incorrect"] += 0.3
    d = rt.paired_delta(better, base, "test", n_boot=300, seed=0)
    assert d["delta"] > 0 and d["delta_lo"] > 0
    perm = base.copy()
    for cls in rt.CLASSES:   # scores permuted within class: no signal change in expectation
        m = perm.true == cls
        perm.loc[m, "p_incorrect"] = rng.permutation(perm.loc[m, "p_incorrect"].to_numpy())
    d0 = rt.paired_delta(perm, perm.assign(p_incorrect=base.p_incorrect.values), "test", 300, 0)
    assert d0["delta_lo"] <= 0 <= d0["delta_hi"]
    with pytest.raises(ValueError, match="no score"):
        rt.paired_delta(better, base.iloc[:5], "test")


def test_paired_delta_resamples_both_models_on_the_same_panoramas(data):
    # Identical scores: every paired resample gives exactly 0; an unpaired resample would not.
    _, df = data
    test = df[df.split.isin(["test", "val"])].assign(split="test")
    a = joined(test, scored(test, np.random.default_rng(11), 0.8))
    d = rt.paired_delta(a, a.copy(), "test", n_boot=200, seed=0)
    assert d["delta"] == 0 and d["delta_lo"] == 0 and d["delta_hi"] == 0


def synth_preds(df, rng):
    """Joined predictions for every arm of country yland (seed 42) and the round-3 baseline."""
    country = country_series(df)
    loco = rt.loco_split(df, country, "yland")
    base = joined(df, scored(df, rng, 1.0))
    g = joined(loco, scored(loco, rng, 1.0))
    cond = g.copy()
    cond.loc[cond.true == "incorrect", "p_incorrect"] = np.clip(cond.loc[cond.true == "incorrect", "p_incorrect"] + 0.25, 0, 1)
    ft = g[g.split.isin(["test", "test_heldout", rt.HELDOUT_COUNTRY])].copy()
    return {rt.tag_global("yland", 42): g, rt.tag_cond("yland", 42): cond, rt.tag_ft("yland", 50, 42): ft,
            rt.tag_condall("country", 42): joined(df, scored(df, rng, 1.2)),
            rt.tag_condall("country", 42) + "-unk": joined(df, scored(df, rng, 1.0))}, {42: base}


def test_report_country_rows_from_synthetic_predictions(data):
    _, df = data
    preds, base = synth_preds(df, np.random.default_rng(4))
    rows = rt.country_rows(preds, base, ["yland"], [42, 43], n_boot=200)
    assert len(rows) == 1
    r = rows[0]
    g = preds[rt.tag_global("yland", 42)]
    ev = g[g.split == rt.HELDOUT_COUNTRY]
    assert r["n_eval"] == len(ev) and r["n_incorrect"] == int((ev.true == "incorrect").sum())
    assert r["cond_auc"] > r["global_auc"] and r["cond_delta"] > 0 and r["verdict"] in ("better", "n too small")
    assert r["global_within_n"] == 2   # gamma and dorado
    from analysis import retrain_report as rr
    w, _ = rr.within_city_auc(ev.true == "incorrect", ev.p_incorrect, ev.city, 10)
    assert r["global_within"] == pytest.approx(w)
    from evaluate import recall_at_precision
    small = ev.head(20)
    pred20 = small.assign(split=rt.HELDOUT_COUNTRY)
    assert rt.recall_at_p(pred20, rt.HELDOUT_COUNTRY, "incorrect") == recall_at_precision(
        (small.true == "incorrect").to_numpy(), small.p_incorrect.to_numpy(), 0.92)[0]
    assert r["drop_test_delta"] is not None and "drop_heldout_delta" in r


def test_curve_rows_order_fractions_and_carry_forgetting(data):
    _, df = data
    preds, _ = synth_preds(df, np.random.default_rng(5))
    meta = {"countries": {"yland": {"fractions": {
        "25": {"train": {"correct": 30, "incorrect": 20}, "skipped": None},
        "50": {"train": {"correct": 60, "incorrect": 40}, "skipped": None},
        "100": {"train": {"correct": 9, "incorrect": 1}, "skipped": "train incorrect 1 < 20"}}}}}
    calib = [{"country": "yland", "seed": 42, "map": "local-50", "verdict": "correct", "gated_n": 70, "coverage": 0.5,
              "precision_raw": 0.95, "wilson_lo": 0.88, "wilson_hi": 0.98}]
    rows = rt.curve_rows(preds, meta, calib, ["yland"], [42], n_boot=100)
    assert [r["pct"] for r in rows] == [0, 25, 50]
    assert rows[0]["auc"] is not None and rows[1]["auc"] is None   # no ft-25 run
    r50 = rows[2]
    assert r50["train_correct"] == 60 and r50["agree_precision_raw"] == 0.95
    assert r50["heldout_delta"] == pytest.approx(0.0) and r50["test_delta"] == pytest.approx(0.0)


def test_condall_rows_pair_with_same_seed_baseline_and_report_per_country(data):
    _, df = data
    preds, base = synth_preds(df, np.random.default_rng(6))
    uid_country = dict(zip(df.label_uid, country_series(df)))
    rows = rt.condall_rows(preds, base, uid_country, ("country", "region"), [42, 43], n_boot=100, min_class=10)
    alls = [r for r in rows if r["scope"] == "all"]
    assert {r["level"] for r in alls} == {"country", "country (scored as unknown)"}   # no region run, no seed 43
    assert all(r["seed"] == 42 for r in rows)
    a = next(r for r in alls if r["level"] == "country")
    assert a["base_auc"] == pytest.approx(rt.auc_on(base[42], "test")) and a["test_delta"] is not None
    scopes = {r["scope"] for r in rows if r["level"] == "country"}
    assert scopes == {"all", "xland", "yland", "wland"}
    small = rt.condall_rows(preds, base, uid_country, ("country",), [42], 50, min_class=10_000)
    assert all(r.get("note") == "n too small" for r in small if r["scope"] != "all")


def test_slice_rows_use_variant_train_counts(data):
    from analysis import slices as sl
    _, df = data
    preds, _ = synth_preds(df, np.random.default_rng(7))
    loco = rt.loco_split(df, country_series(df), "yland")
    ft_train = loco[loco.split == rt.POOL].head(40)
    train = {("cond", "yland"): loco[loco.split == "train"], ("ft-50", "yland"): ft_train}
    rows = rt.slice_rows(preds, train, ["yland"], 42, n_boot=50, min_n=5)
    ft_sev = [r for r in rows if r["arm"] == "ft-50" and r["by"] == "severity"]
    expected = sl.with_slice(ft_train, "severity").groupby("slice").size()
    assert ft_sev and all(r["train_n"] == int(expected.get(r["slice"], 0)) for r in ft_sev)
    assert all(r["thin"] for r in ft_sev)   # 40 rows < THIN_TRAIN
    assert {r["arm"] for r in rows} == {"cond", "ft-50"} and {r["by"] for r in rows} == {"severity", "tag"}


def test_seed_rows_pair_same_seed_and_summarise(data):
    _, df = data
    preds, _ = synth_preds(df, np.random.default_rng(8))
    preds2, _ = synth_preds(df, np.random.default_rng(9))
    preds.update({k.replace("-s42", "-s43"): v for k, v in preds2.items()})
    rows = rt.seed_rows(preds, ["yland"], [42, 43])
    assert {(r["variant"], r["seed"]) for r in rows} == {("global|yland", 42), ("cond|yland", 42),
                                                        ("global|yland", 43), ("cond|yland", 43)}
    cond = [r for r in rows if r["variant"] == "cond|yland"]
    assert all(r["heldout_delta"] > 0 for r in cond)
    s = {x["variant"]: x for x in cl.seed_summary(rows)}
    assert s["cond|yland"]["n_seeds"] == 2 and s["cond|yland"]["heldout_delta_positive"] == 2


def test_judge_and_redact():
    assert rt.judge({"delta": -0.05, "delta_lo": -0.08, "delta_hi": -0.01}, 200) == "worse"
    assert rt.judge({"delta": 0.0, "delta_lo": -0.02, "delta_hi": 0.02}, 200) == "no clear difference"
    assert rt.judge({"delta": 0.05, "delta_lo": 0.01, "delta_hi": 0.09}, 50) == "n too small"
    assert rt.judge(None, 500) == "n too small"
    with pytest.raises(ValueError, match="non-public"):
        rt.redact({"x": "trained on Omegaville crops"}, {"Omegaville"})
    rt.redact({"x": "omegavilles"}, {"omegaville"})   # whole tokens only


# ---------------------------------------------------------------------------------------------------------------------
# report: end to end
# ---------------------------------------------------------------------------------------------------------------------

def doc_template(path):
    blocks = rt.render_blocks(None)
    text = "# Doc\n\nIntro stays.\n\n" + "\n\n".join(
        rt.MARK_BEGIN.format(name=b) + "\n\n" + blocks[b] + "\n\n" + rt.MARK_END.format(name=b) for b in rt.BLOCKS) + "\n"
    path.write_text(text)
    return text


def write_run(data_dir, df):
    """Predictions for yland seed 42 under the eval dir, a round-3 baseline, and calibration records."""
    preds, base = synth_preds(df, np.random.default_rng(10))
    for tag, p in preds.items():
        path = rt.pred_path(str(data_dir), LT, tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        p[["label_uid", "split", "true", "p_correct", "p_incorrect"]].to_csv(path, index=False)
    base[42][["label_uid", "split", "true", "p_correct", "p_incorrect"]].to_csv(data_dir / LT / "predictions.csv",
                                                                               index=False)
    (data_dir / "priors.json").write_text(json.dumps({LT: {"correct": 900, "incorrect": 100}}))
    rt.main(["calibrate", "--data-dir", str(data_dir), "--label-type", LT, "--priors", str(data_dir / "priors.json"),
             "--cities-json", str(data_dir / "cities.json")])
    (data_dir / "region_jobs.txt").write_text("Toy yland global=123456 cond=123457 ft=123458\n")


def test_report_names_public_countries_and_letters_private_only_ones_and_redacts(data, tmp_path):
    data_dir, df = data
    run_variants(data_dir)
    write_run(data_dir, df)
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    doc_template(doc)
    args = ["report", "--data-dir", str(data_dir), "--label-types", LT, "--cities-json", str(data_dir / "cities.json"),
            "--public-cities", str(data_dir / "cities.json"), "--seeds", "42", "--n-boot", "50", "--out", str(out),
            "--doc", str(doc), "--jobs-file", str(data_dir / "region_jobs.txt")]
    assert rt.main(args) == 0
    text = "".join(p.read_text() for p in out.iterdir()) + doc.read_text()
    assert "yland" in text and "xland" in text and "private country A" in text
    for term in ("dorado", "omega", "zland", "Omegaville", "Doradoville"):
        assert not __import__("re").search(r"(?<![a-z0-9])" + term.lower() + r"(?![a-z0-9])", text.lower()), term
    res = json.loads((out / "results.json").read_text())
    assert res["provenance"]["job_ids"] == ["123456", "123457", "123458"]
    assert any(k.startswith("predictions:global-yland-s42") for k in res["provenance"]["inputs_sha256"][LT])
    # Visibility drifts: when xland's deployments turn private, the next report letters it instead of naming it.
    listing = json.loads((data_dir / "cities.json").read_text())
    for c in listing["cities"][:2]:
        c["visibility"] = "private"
    (data_dir / "naming.json").write_text(json.dumps(listing))
    i = args.index("--public-cities")
    args2 = args[:i + 1] + [str(data_dir / "naming.json")] + args[i + 2:]
    args2[args2.index("--out") + 1] = str(tmp_path / "out2")
    assert rt.main(args2) == 0
    text2 = (tmp_path / "out2" / "results.json").read_text() + doc.read_text()
    assert "xland" not in text2 and "private country B" in text2


def test_report_writes_markers_and_json_and_provenance(data, tmp_path):
    data_dir, df = data
    run_variants(data_dir)
    write_run(data_dir, df)
    out, doc = tmp_path / "out", tmp_path / "doc.md"
    template = doc_template(doc)
    check = ["report", "--check", "--out", str(out), "--doc", str(doc)]
    assert rt.main(check) == 0            # not yet run: placeholders match
    assert rt.main(["report", "--data-dir", str(data_dir), "--label-types", LT, "--cities-json",
                    str(data_dir / "cities.json"), "--public-cities", str(data_dir / "cities.json"), "--seeds", "42",
                    "--n-boot", "50", "--out", str(out), "--doc", str(doc)]) == 0
    filled = doc.read_text()
    assert filled != template and filled.startswith("# Doc\n\nIntro stays.\n\n")
    assert "**Status: run.**" in filled and rt.NOT_RUN not in filled
    for name in ("results.json", "results.md", "slices.md", "seeds.md", "provenance.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "provenance.json").read_text())["n_boot"] == 50
    assert rt.main(check) == 0            # filled and in sync
    doc.write_text(filled.replace("| Toy |", "| Toy  |", 1))
    assert rt.main(check) == 1            # one byte of drift fails
    doc.write_text(template)
    assert rt.main(check) == 1            # results exist but the doc still says not yet run


def test_committed_doc_passes_check():
    root = os.path.join(os.path.dirname(__file__), "..")
    assert rt.check_doc(os.path.join(root, "docs", "region-transfer-2026-11.md"),
                        os.path.join(root, "docs", "region-transfer-2026-11")) == []


def test_main_cli_parses_every_stage():
    p = rt.build_parser()
    for argv in (["variants", "--split", "s", "--cities-json", "c", "--out", "o"], ["list", "--variants", "v"],
                 ["predict", "--split-file", "s", "--data-dir", "d", "--label-type", "t", "--out", "o"],
                 ["calibrate", "--data-dir", "d", "--label-type", "t", "--priors", "p", "--cities-json", "c"],
                 ["report", "--check"]):
        assert p.parse_args(argv).func is not None
    with pytest.raises(SystemExit):
        rt.main(["report", "--out", "x", "--doc", "y"])   # neither data nor --check


def test_tags_and_paths():
    assert rt.tag_ft("mexico", 25, 42) == "ft-mexico-f25-s42"
    assert rt.pred_path("/d", "CurbRamp", "x").endswith("/d/CurbRamp/region/eval/CurbRamp/predictions_x.csv")
    assert rt.rename_hash_keys({"predictions:cond-zland-s42": "h"}, {"zland": "private country A"}) == \
        {"predictions:cond-private_country_A-s42": "h"}


def test_doctests():
    assert doctest.testmod(rt).failed == 0
