"""analysis/confident_learning.py: folds, the stricter rules, the confident joint, the variants and the report."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import confident_learning as cl  # noqa: E402
import fine_tune_classifier as ft  # noqa: E402

COLS = ["label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "class_source", "author",
        "human_agree", "human_disagree", "severity", "tags", "latitude", "longitude", "hard", "crop_path", "status",
        "split"]


def make_split(tmp_path, n_panos=40, seed=0):
    """A split.csv like split_data.py writes: train/val/test/test_heldout, 1-3 labels per panorama, vote counts."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for p in range(n_panos):
        city = ["alpha", "beta", "gamma"][p % 3]
        split = "test_heldout" if city == "gamma" else ["train", "train", "train", "val", "test"][p % 5]
        for _ in range(int(rng.integers(1, 4))):
            cls = "incorrect" if rng.random() < 0.45 else "correct"
            if cls == "incorrect":
                agree, disagree = int(rng.integers(0, 2)), int(rng.integers(2, 6))
            else:
                agree, disagree = int(rng.integers(3, 7)), int(rng.integers(0, 2))
            expert = rng.random() < 0.05
            rows.append({"label_uid": f"{city}:{i}", "city": city, "label_id": str(i), "pano_id": f"pano{p:03d}",
                         "label_type": "Toy", "label_class": cls, "class_source": "expert" if expert else "margin",
                         "author": "human" if rng.random() < 0.9 else "ai",
                         "human_agree": "0" if expert else str(agree), "human_disagree": "0" if expert else str(disagree),
                         "severity": str(int(rng.integers(1, 6))), "tags": "[]",
                         "latitude": f"{47.6 + p * 0.001:.6f}", "longitude": f"{-122.3 + (i % 2) * 0.00001:.6f}",
                         "hard": "0", "crop_path": f"{cls}/{city}_{i}.webp", "status": "ok", "split": split})
            i += 1
    df = pd.DataFrame(rows, columns=COLS).sort_values("label_uid").reset_index(drop=True)
    base = tmp_path / "Toy"
    base.mkdir(exist_ok=True)
    for cls in cl.CLASSES:
        (base / cls).mkdir(exist_ok=True)
    for _, r in df.iterrows():
        (base / r.crop_path).write_bytes(b"x")
    path = base / "split.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    return path, df


@pytest.fixture
def toy(tmp_path):
    return make_split(tmp_path)


def test_folds_keep_panoramas_together_balance_and_leave_eval_rows_alone(toy):
    path, df = toy
    out = path.parent / "cl"
    assert cl.main(["folds", "--split", str(path), "--folds", "3", "--seed", "1", "--merge-radius-m", "0",
                    "--out", str(out)]) == 0
    folds = pd.read_csv(out / "folds.csv")
    train = df[df.split == "train"]
    assert set(folds.label_uid) == set(train.label_uid)
    joined = train.merge(folds, on="label_uid")
    assert (joined.groupby("pano_id").fold.nunique() == 1).all()
    sizes = joined.fold.value_counts()
    # greedy fill per class-ratio bin: each bin is off by at most one group, so the imbalance is bounded by that
    assert sizes.max() - sizes.min() <= 5 * joined.groupby("pano_id").size().max()
    for k in range(3):
        v = pd.read_csv(out / f"fold{k}" / "split.csv", dtype=str, keep_default_na=False)
        assert set(v.loc[v.split == cl.FOLD_OUT, "label_uid"]) == set(folds.loc[folds.fold == k, "label_uid"])
        assert (v[v.split != "train"][v.split != cl.FOLD_OUT].split.isin(["val", "test", "test_heldout"])).all()
        # val/test/test_heldout rows byte-identical to the source
        base_eval = df[df.split != "train"].sort_values("label_uid").reset_index(drop=True)
        v_eval = v[v.label_uid.isin(base_eval.label_uid)].sort_values("label_uid").reset_index(drop=True)
        assert base_eval.astype(str).equals(v_eval[base_eval.columns].astype(str))
    meta = json.load(open(out / "folds.json"))
    assert meta["folds"] == 3 and meta["seed"] == 1 and len(meta["rows_per_fold"]) == 3


def test_folds_are_deterministic_and_seed_dependent(toy):
    path, _ = toy
    a = path.parent / "a"
    b = path.parent / "b"
    c = path.parent / "c"
    for out, seed in ((a, "7"), (b, "7"), (c, "8")):
        cl.main(["folds", "--split", str(path), "--folds", "4", "--seed", seed, "--merge-radius-m", "0",
                 "--out", str(out)])
    assert (a / "folds.csv").read_bytes() == (b / "folds.csv").read_bytes()
    assert (a / "folds.csv").read_bytes() != (c / "folds.csv").read_bytes()
    assert (a / "fold0" / "split.csv").read_bytes() == (b / "fold0" / "split.csv").read_bytes()


def test_folds_merge_nearby_panoramas_with_coordinates(toy):
    path, df = toy
    out = path.parent / "cl"
    # The toy panoramas sit ~111 m apart in latitude (train ones up to 333 m apart); a 2 km radius chains them all
    # into one group, which then cannot be split: every train row lands in one fold.
    assert cl.main(["folds", "--split", str(path), "--folds", "2", "--merge-radius-m", "2000",
                    "--out", str(out)]) == 0
    folds = pd.read_csv(out / "folds.csv")
    assert folds.fold.nunique() == 1
    # Without coordinates in the manifest, a radius needs --rawlabels.
    bare = df.drop(columns=["latitude", "longitude"])
    bare_path = path.parent / "bare.csv"
    bare.to_csv(bare_path, index=False)
    with pytest.raises(ValueError):
        cl.main(["folds", "--split", str(bare_path), "--folds", "2", "--merge-radius-m", "3", "--out", str(out)])


def test_confident_joint_and_flags_by_hand():
    y = np.array(["correct"] * 4 + ["incorrect"] * 4)
    p_inc = np.array([0.05, 0.10, 0.90, 0.30, 0.95, 0.85, 0.20, 0.60])
    p = np.column_stack([1 - p_inc, p_inc])
    joint, t = cl.confident_joint(y, p)
    # t_correct = mean p_correct over given-correct = (0.95+0.90+0.10+0.70)/4 = 0.6625
    # t_incorrect = mean p_incorrect over given-incorrect = (0.95+0.85+0.20+0.60)/4 = 0.65
    assert t == pytest.approx([0.6625, 0.65])
    # given correct: rows 0,1,3 -> correct (p_cor .95/.90/.70 >= .6625); row 2 (p_inc .90 >= .65) -> incorrect
    # given incorrect: rows 4,5 -> incorrect; row 6 (p_cor .80 >= .6625) -> correct; row 7 (.60 < .65, .40 < .6625) none
    assert joint.tolist() == [[3, 1], [1, 2]]
    flagged, likely = cl.flag_rows(y, p)
    assert flagged.tolist() == [False, False, True, False, False, False, True, False]
    assert likely.tolist() == ["correct", "correct", "incorrect", "correct", "incorrect", "incorrect", "correct",
                               "incorrect"]


def test_flag_takes_the_highest_scoring_rows_per_cell_not_just_any_contradiction():
    # Three given-correct rows have argmax incorrect, but only two clear t_incorrect (0.9): the cell says two are
    # wrong, and the two with the highest p_incorrect are the ones flagged.
    y = np.array(["correct", "correct", "correct", "correct", "incorrect", "incorrect"])
    p_inc = np.array([0.95, 0.90, 0.80, 0.05, 0.9, 0.9])
    p = np.column_stack([1 - p_inc, p_inc])
    joint, t = cl.confident_joint(y, p)
    assert t == pytest.approx([0.325, 0.9])
    assert joint.tolist() == [[1, 2], [0, 2]]
    flagged, likely = cl.flag_rows(y, p)
    assert flagged.tolist() == [True, True, False, False, False, False]
    assert likely.tolist() == ["incorrect", "incorrect", "correct", "correct", "incorrect", "incorrect"]


def test_flag_command_writes_noisy_targets_and_variants(toy):
    path, df = toy
    out = path.parent / "cl"
    cl.main(["folds", "--split", str(path), "--folds", "2", "--merge-radius-m", "0", "--out", str(out)])
    folds = pd.read_csv(out / "folds.csv", dtype={"label_uid": str})
    train = df[df.split == "train"].merge(folds, on="label_uid")
    rng = np.random.default_rng(0)
    for k in (0, 1):
        part = train[train.fold == k]
        # the model mostly agrees with the target, with a few confident contradictions
        p_inc = np.where(part.label_class == "incorrect", 0.9, 0.1) + rng.normal(0, 0.05, len(part))
        p_inc[:2] = 1 - p_inc[:2]
        p_inc = np.clip(p_inc, 0.001, 0.999)
        pd.DataFrame({"label_uid": part.label_uid, "fold": k, "label_class": part.label_class,
                      "p_correct": 1 - p_inc, "p_incorrect": p_inc}).to_csv(out / f"oof_fold{k}.csv", index=False)
    assert cl.main(["flag", "--split", str(path), "--oof", str(out / "oof_fold*.csv"), "--out", str(out)]) == 0
    noisy = pd.read_csv(out / "noisy_targets.csv", dtype={"label_uid": str})
    assert list(noisy.columns) == cl.NOISY_COLUMNS
    assert len(noisy) == (df.split == "train").sum()
    assert noisy.flagged.sum() >= 2
    summary = json.load(open(out / "flag_summary.json"))
    assert summary["flagged"] == int(noisy.flagged.sum())
    assert set(summary["confident_joint"]) == set(cl.CLASSES)
    flagged = set(noisy.loc[noisy.flagged == 1, "label_uid"])
    drop = pd.read_csv(out / "cl-drop" / "split.csv", dtype=str, keep_default_na=False)
    flip = pd.read_csv(out / "cl-flip" / "split.csv", dtype=str, keep_default_na=False)
    assert set(drop.loc[drop.split == cl.TRAIN_DROPPED, "label_uid"]) == flagged
    assert (flip.split == df.set_index("label_uid").loc[flip.label_uid, "split"].to_numpy()).all()
    orig = df.set_index("label_uid").label_class
    changed = flip[flip.label_class != orig.loc[flip.label_uid].to_numpy()]
    assert set(changed.label_uid) == flagged
    assert (changed.class_source == "cl_flip").all()
    assert (flip.crop_path == df.set_index("label_uid").loc[flip.label_uid, "crop_path"].to_numpy()).all()


def test_flag_refuses_incomplete_or_duplicated_oof(toy):
    path, df = toy
    out = path.parent / "cl"
    out.mkdir()
    train = df[df.split == "train"]
    half = train.iloc[: len(train) // 2]
    pd.DataFrame({"label_uid": half.label_uid, "fold": 0, "label_class": half.label_class,
                  "p_correct": 0.5, "p_incorrect": 0.5}).to_csv(out / "oof_fold0.csv", index=False)
    with pytest.raises(SystemExit, match="unscored"):
        cl.main(["flag", "--split", str(path), "--oof", str(out / "oof_fold0.csv"), "--out", str(out)])
    pd.DataFrame({"label_uid": train.label_uid, "fold": 1, "label_class": train.label_class,
                  "p_correct": 0.5, "p_incorrect": 0.5}).to_csv(out / "oof_fold1.csv", index=False)
    with pytest.raises(SystemExit, match="more than one"):
        cl.main(["flag", "--split", str(path), "--oof", str(out / "oof_fold*.csv"), "--out", str(out)])


def test_rules_by_hand_and_expert_rows_pass():
    d = pd.DataFrame({"label_class": ["incorrect", "incorrect", "incorrect", "correct", "correct", "correct"],
                      "class_source": ["margin", "margin", "expert", "margin", "margin", "margin"],
                      "human_agree": ["0", "1", "0", "3", "4", "5"], "human_disagree": ["2", "4", "0", "0", "1", "0"]})
    assert cl.passes_rule(d, "symmetric").tolist() == [False, True, True, True, True, True]
    assert cl.passes_rule(d, "unanimous3").tolist() == [False, False, True, True, False, True]
    with pytest.raises(ValueError):
        cl.passes_rule(d, "margin")


def test_rules_command_drops_only_train_rows_and_records_counts(toy):
    path, df = toy
    out = path.parent / "cl"
    assert cl.main(["rules", "--split", str(path), "--out", str(out)]) == 0
    for rule in cl.RULES:
        v = pd.read_csv(out / rule / "split.csv", dtype=str, keep_default_na=False)
        merged = df.merge(v[["label_uid", "split", "label_class"]], on="label_uid", suffixes=("", "_v"))
        non_train = merged[merged.split != "train"]
        assert (non_train.split == non_train.split_v).all()
        assert (merged.label_class == merged.label_class_v).all()
        dropped = merged[(merged.split == "train") & (merged.split_v == cl.TRAIN_DROPPED)]
        kept = merged[(merged.split == "train") & (merged.split_v == "train")]
        assert (~cl.passes_rule(dropped, rule)).all()
        assert cl.passes_rule(kept, rule).all()
        meta = json.load(open(out / rule / "rule.json"))
        assert meta["dropped"] == len(dropped)
        assert sum(meta["train_after"].values()) == len(kept)
    # unanimous3 is at least as strict as symmetric
    sym = pd.read_csv(out / "symmetric" / "split.csv", dtype=str)
    una = pd.read_csv(out / "unanimous3" / "split.csv", dtype=str)
    assert (una.split == cl.TRAIN_DROPPED).sum() >= (sym.split == cl.TRAIN_DROPPED).sum()


def test_trainer_split_flag_reads_the_variant_and_keys_on_its_hash(toy, tmp_path):
    path, df = toy
    out = path.parent / "cl"
    cl.main(["rules", "--split", str(path), "--rule", "symmetric", "--out", str(out)])
    variant = out / "symmetric" / "split.csv"
    train_df, val_df = ft.load_split_frames("Toy", str(tmp_path), split=str(variant))
    v = pd.read_csv(variant)
    assert len(train_df) == (v.split == "train").sum() < (df.split == "train").sum()
    assert len(val_df) == (df.split == "val").sum()
    assert all(os.path.isabs(p) and os.path.exists(p) for p in train_df.image)  # crops resolve under <data>/Toy/

    class Args:
        seed, epochs, best_metric, target, vote_weights = 42, 25, "macro_f1", "hard", None

    a = Args()
    key_default = ft.run_key("Toy", str(tmp_path), a)
    a.split = str(variant)
    key_variant = ft.run_key("Toy", str(tmp_path), a)
    assert key_default["split_sha256"] != key_variant["split_sha256"]
    assert key_variant["split_sha256"] == cl.sha256_of(str(variant))


@pytest.mark.parametrize("argv, message", [
    (["--label_type", "Toy", "--split", "x.csv"], "--output-dir"),
    (["--label_type", "Toy", "--split", "missing.csv", "--output-dir", "o"], "not found"),
])
def test_trainer_split_flag_checks(argv, message, capsys):
    with pytest.raises(SystemExit):
        ft.main(argv)
    assert message in capsys.readouterr().err


def test_report_tables_and_expert_aggregates(toy, tmp_path):
    path, df = toy
    out = path.parent / "cl"
    cl.main(["rules", "--split", str(path), "--out", str(out)])

    def metrics(auc, heldout, r92):
        return {"test": {"auc": auc, "n": 10, "recall_at_precision": {"incorrect": {"0.92": {"recall": r92}},
                                                                       "correct": {"0.92": {"recall": 0.9}}}},
                "test_heldout": {"auc": heldout, "recall_at_precision": {"incorrect": {"0.92": {"recall": 0.1}}}},
                "hard": {"auc": 0.5, "n": 7},
                "per_city": {"test": {"min_city_n": 50, "cities": {
                                          "alpha": {"auc": 0.8, "n_by_class": {"correct": 10, "incorrect": 10}},
                                          "beta": {"auc": 0.9, "n_by_class": {"correct": 30, "incorrect": 10}}}},
                             "test_heldout": {"min_city_n": 50, "cities": {
                                          "gamma": {"auc": 0.7, "n_by_class": {"correct": 5, "incorrect": 5}}}}}}
    json.dump(metrics(0.95, 0.9, 0.3), open(path.parent / "metrics.json", "w"))
    json.dump(metrics(0.96, 0.91, 0.35), open(path.parent / "metrics_symmetric.json", "w"))
    # predictions for the baseline only: its within-city AUC then comes from them, symmetric's from the per_city block
    rng = np.random.default_rng(3)
    ev = df[df.split != "train"]
    pred = pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class,
                         "p_incorrect": np.where(ev.label_class == "incorrect", rng.uniform(0.4, 1, len(ev)),
                                                 rng.uniform(0, 0.6, len(ev)))})
    pred["p_correct"] = 1 - pred.p_incorrect
    pred.to_csv(path.parent / "predictions.csv", index=False)
    train = df[df.split == "train"]
    gold = pd.DataFrame({"label_uid": train.label_uid.iloc[:6],
                         "verdict": ["correct", "incorrect", "correct", "incorrect", "cant_tell", "correct"]})
    # a second rater who agrees on the first item and disagrees on the second: the second becomes unrated
    gold = pd.concat([gold, pd.DataFrame({"label_uid": train.label_uid.iloc[:2], "verdict": ["correct", "correct"]})])
    gold_path = tmp_path / "gold.csv"
    gold.to_csv(gold_path, index=False)
    json.dump({"by_city": [{"city": "alpha", "n": 10, "flagged": 1, "share": 0.1},
                           {"city": "beta", "n": 20, "flagged": 4, "share": 0.2},
                           {"city": "delta", "n": 10, "flagged": 5, "share": 0.5}],
               "source_split": "/private/path", "flagged": 10}, open(out / "flag_summary.json", "w"))
    doc = tmp_path / "doc"
    assert cl.main(["report", "--data-dir", str(tmp_path), "--label-types", "Toy", "--variants",
                    "symmetric,unanimous3", "--gold", str(gold_path), "--public-cities", "alpha",
                    "--n-boot", "0", "--within-min-n", "2", "--out", str(doc)]) == 0
    text = (doc / "variants.md").read_text()
    assert "| baseline |" in text and "| symmetric |" in text and "| unanimous3 |" in text
    assert "not run" in text  # unanimous3 has no metrics file
    assert "0.875 (2)" in text  # symmetric's within-city test AUC from the per_city block: (0.8*100 + 0.9*300) / 400
    data = json.load(open(doc / "variants.json"))
    rows = {r["variant"]: r for r in data["Toy"]["metrics"]}
    assert rows["symmetric"]["incorrect_recall_p92_test"] == 0.35
    assert rows["symmetric"]["within_city_source"].startswith("metrics.json")
    assert rows["baseline"]["within_city_source"].startswith("predictions")
    assert rows["baseline"]["cities_test"] == 2 and rows["baseline"]["cities_heldout"] == 1  # alpha, beta; gamma
    from analysis import retrain_report as rr
    p = pred[pred.split == "test"].merge(df[["label_uid", "city"]], on="label_uid")
    wc, _ = rr.within_city_auc(p.true == "incorrect", p.p_incorrect, p.city, min_n=2)
    assert rows["baseline"]["within_city_test"] == pytest.approx(wc)
    # held-out table: gamma is not public and not a US city id, so it is lettered as in the round-3 report
    held = data["Toy"]["heldout"]
    assert [h["deployment"] for h in held] == ["private city A (outside the US)"] and "gamma" not in text
    assert held[0]["n"] == int((ev.split == "test_heldout").sum()) and 0 < held[0]["baseline"] <= 1
    assert "symmetric" not in held[0]  # no predictions for it
    assert data["Toy"]["slices"] == [] and not (doc / "slices.md").exists()  # no variant has predictions
    acc = {r["variant"]: r for r in data["Toy"]["train"]}
    assert acc["baseline"]["train"] == len(train) and acc["symmetric"]["dropped"] > 0
    exp = {r["variant"]: r for r in data["Toy"]["expert"]}
    assert exp["symmetric"]["n_rated_train"] == 4  # the cant_tell row and the rater disagreement are not rated
    assert exp["symmetric"]["touched"] + exp["symmetric"]["untouched"] == 4
    assert "label_uid" not in text  # no ids in the published table
    flags = data["Toy"]["flag_summary"]
    assert "source_split" not in flags
    assert [r["city"] for r in flags["by_city"]] == ["other (private deployments)", "alpha"]
    assert flags["by_city"][0] == {"city": "other (private deployments)", "n": 30, "flagged": 9, "share": 0.3}
    assert data["_public_cities"] == ["alpha"]


def test_rule_priors_count_human_margin_labels_under_each_rule(tmp_path):
    import data_generation as dg

    def votes(agree, disagree, user="u1"):
        return json.dumps([{"user_id": user, "validation": "Agree", "validator_type": "Human"}] * agree
                          + [{"user_id": user, "validation": "Disagree", "validator_type": "Human"}] * disagree)

    labels = pd.DataFrame({
        "label_type": ["CurbRamp"] * 6, "label_id": [1, 2, 3, 4, 5, 6], "city": ["a"] * 6,
        "validations": [votes(3, 0), votes(4, 1), votes(0, 2), votes(0, 3), votes(1, 4), votes(1, 1)],
        "pano_source": ["gsv"] * 6,
        "pano_x": [10] * 6, "pano_y": [10] * 6, "pano_width": [100] * 6, "pano_height": [50] * 6,
        "user_id": ["h"] * 6})
    eligible, _ = dg.eligible_labels(labels, "CurbRamp")
    assert len(eligible) == 5  # the 1:1 label is contested
    out = cl.rule_priors(labels, ("CurbRamp",))
    assert out["margin"]["CurbRamp"] == {"correct": 2, "incorrect": 3, "share_incorrect": 0.6}
    # symmetric: incorrect needs net Disagree > 2 -> (0,3) and (1,4) only
    assert out["symmetric"]["CurbRamp"]["incorrect"] == 2 and out["symmetric"]["CurbRamp"]["correct"] == 2
    # unanimous3: (3,0) and (0,3) only
    assert out["unanimous3"]["CurbRamp"] == {"correct": 1, "incorrect": 1, "share_incorrect": 0.5}


def test_gold_adjudication_keeps_only_agreeing_raters():
    gold = pd.DataFrame({"label_uid": ["x:1", "x:1", "x:2", "x:2", "x:3", "x:4", "x:4"],
                         "verdict": ["correct", "correct", "correct", "incorrect", "incorrect", "cant_tell", "cant_tell"]})
    out = cl.adjudicate_gold(gold).set_index("label_uid").verdict
    assert out.to_dict() == {"x:1": "correct", "x:2": "cant_tell", "x:3": "incorrect", "x:4": "cant_tell"}


def test_report_slices_against_baseline(tmp_path):
    path, df = make_split(tmp_path, n_panos=150)  # enough held-out labels of each class for a slice interval
    out = path.parent / "cl"
    cl.main(["rules", "--split", str(path), "--out", str(out)])
    rng = np.random.default_rng(5)
    ev = df[df.split != "train"].copy()
    for name, shift in (("", 0.0), ("_symmetric", 0.1)):
        pred = pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class})
        pred["p_incorrect"] = np.clip(np.where(ev.label_class == "incorrect", rng.uniform(0.3, 1, len(ev)) + shift,
                                               rng.uniform(0, 0.7, len(ev)) - shift), 0, 1)
        pred["p_correct"] = 1 - pred.p_incorrect
        pred.to_csv(path.parent / f"predictions{name}.csv", index=False)
        json.dump({"test": {"auc": 0.9, "n": 1, "recall_at_precision": {}}, "test_heldout": {"auc": 0.8},
                   "hard": {}}, open(path.parent / f"metrics{name}.json", "w"))
    doc = tmp_path / "doc"
    assert cl.main(["report", "--data-dir", str(tmp_path), "--label-types", "Toy", "--variants", "symmetric",
                    "--public-cities", "alpha,beta,gamma", "--n-boot", "20", "--slice-min-n", "5",
                    "--out", str(doc)]) == 0
    data = json.load(open(doc / "variants.json"))
    held = {h["deployment"]: h for h in data["Toy"]["heldout"]}
    assert set(held) == {"gamma"} and held["gamma"]["symmetric"] >= held["gamma"]["baseline"]  # public: named
    slices = data["Toy"]["slices"]
    assert {(r["split"], r["by"]) for r in slices} == {("test", "severity"), ("test", "tag"),
                                                        ("test_heldout", "severity"), ("test_heldout", "tag")}
    tag_rows = [r for r in slices if r["by"] == "tag" and r["split"] == "test_heldout"]
    assert [r["slice"] for r in tag_rows] == ["(none)"]  # the toy has no tags
    assert tag_rows[0]["n"] == int((ev.split == "test_heldout").sum())
    assert tag_rows[0]["delta"] == pytest.approx(tag_rows[0]["auc_a"] - tag_rows[0]["auc_b"])
    assert tag_rows[0]["delta_lo"] <= tag_rows[0]["delta"] <= tag_rows[0]["delta_hi"]
    md = (doc / "slices.md").read_text()
    assert "### Toy / test / by severity" in md and "symmetric Δ [95% CI]" in md
    rows = {r["variant"]: r for r in data["Toy"]["metrics"]}
    assert rows["symmetric"]["within_city_source"].startswith("predictions")
    assert "test_delta" not in rows["baseline"]
    d = rows["symmetric"]
    assert d["heldout_delta_lo"] <= d["heldout_delta"] <= d["heldout_delta_hi"]
    assert d["test_delta_lo"] <= d["test_delta"] <= d["test_delta_hi"]
    text = (doc / "variants.md").read_text()
    assert "Held-out deployments" in text and "Δ held-out [95% CI]" in text


def test_report_seed_replicates_pair_each_seed_with_its_own_baseline(tmp_path):
    path, df = make_split(tmp_path, n_panos=150)
    out = path.parent / "cl"
    cl.main(["rules", "--split", str(path), "--out", str(out)])
    ev = df[df.split != "train"].copy()
    rng = np.random.default_rng(7)

    def write(name, shift, auc):
        pred = pd.DataFrame({"label_uid": ev.label_uid, "split": ev.split, "true": ev.label_class})
        pred["p_incorrect"] = np.clip(np.where(ev.label_class == "incorrect", rng.uniform(0.3, 1, len(ev)) + shift,
                                               rng.uniform(0, 0.7, len(ev)) - shift), 0, 1)
        pred["p_correct"] = 1 - pred.p_incorrect
        suffix = "" if name == "baseline" else f"_{name}"
        pred.to_csv(path.parent / f"predictions{suffix}.csv", index=False)
        json.dump({"test": {"auc": auc, "recall_at_precision": {}}, "test_heldout": {"auc": auc - 0.1}, "hard": {}},
                  open(path.parent / f"metrics{suffix}.json", "w"))

    # the chain's seed-42 runs, two replicate seeds of both, and a cl-flip replicate without a seed-42 run
    write("baseline", 0.0, 0.90); write("symmetric", 0.1, 0.92)
    write("baseline-s43", 0.0, 0.91); write("symmetric-s43", 0.1, 0.93)
    write("baseline-s44", 0.0, 0.89); write("symmetric-s44", 0.1, 0.91)
    write("unanimous3-s43", 0.05, 0.95)
    runs = cl.seed_runs(str(tmp_path), "Toy")
    assert runs == {("baseline", 42): "baseline", ("baseline", 43): "baseline-s43", ("baseline", 44): "baseline-s44",
                    ("symmetric", 42): "symmetric", ("symmetric", 43): "symmetric-s43",
                    ("symmetric", 44): "symmetric-s44", ("unanimous3", 43): "unanimous3-s43"}
    assert cl.seed_runs(str(tmp_path), "Nope") == {}
    doc = tmp_path / "doc"
    assert cl.main(["report", "--data-dir", str(tmp_path), "--label-types", "Toy", "--variants", "symmetric",
                    "--public-cities", "alpha,beta,gamma", "--n-boot", "20", "--slice-min-n", "5",
                    "--out", str(doc)]) == 0
    data = json.load(open(doc / "variants.json"))["Toy"]["seeds"]
    rows = {(r["variant"], r["seed"]): r for r in data["rows"]}
    assert [r["variant"] for r in data["rows"]][:3] == ["baseline"] * 3  # baseline first, then by variant and seed
    assert rows[("symmetric", 43)]["test_auc"] == 0.93 and rows[("symmetric", 43)]["run"] == "symmetric-s43"
    for seed in (42, 43, 44):  # each replicate is paired with the baseline of its own seed
        r = rows[("symmetric", seed)]
        assert r["heldout_delta_lo"] <= r["heldout_delta"] <= r["heldout_delta_hi"]
        assert r["heldout_delta"] > 0 and r["test_delta"] > 0
    assert "heldout_delta" not in rows[("baseline", 43)]
    assert "heldout_delta" in rows[("unanimous3", 43)]  # no seed-42 run, but baseline-s43 exists to pair with
    summary = {s["variant"]: s for s in data["summary"]}
    assert summary["baseline"]["seeds"] == [42, 43, 44] and summary["baseline"]["n_deltas"] == 0
    assert summary["baseline"]["test_auc_mean"] == pytest.approx(0.90) and summary["baseline"]["test_auc_sd"] == pytest.approx(0.01)
    assert summary["symmetric"]["n_deltas"] == 3 and summary["symmetric"]["heldout_delta_positive"] == 3
    assert summary["symmetric"]["heldout_delta_min"] <= summary["symmetric"]["heldout_delta_mean"] <= summary["symmetric"]["heldout_delta_max"]
    assert summary["unanimous3"]["n_seeds"] == 1 and summary["unanimous3"]["test_auc_sd"] is None
    md = (doc / "seeds.md").read_text()
    assert "### Toy" in md and "| symmetric | 43 |" in md and "3/3 > 0" in md
    assert "| unanimous3 | 43 |" in md and "| unanimous3 | 43 | 0.950 |" in md and "1/1 > 0" in md
    assert summary["symmetric"]["n_deltas_test"] == summary["symmetric"]["n_deltas_heldout"] == 3
    # a replicate without predictions (metrics only) gets no Δ, and the table says why
    json.dump({"test": {"auc": 0.5}, "test_heldout": {"auc": 0.5}, "hard": {}},
              open(path.parent / "metrics_symmetric-s45.json", "w"))
    assert cl.main(["report", "--data-dir", str(tmp_path), "--label-types", "Toy", "--variants", "symmetric",
                    "--public-cities", "alpha,beta,gamma", "--n-boot", "0", "--slice-min-n", "5",
                    "--out", str(doc)]) == 0
    data = json.load(open(doc / "variants.json"))["Toy"]["seeds"]
    r45 = next(r for r in data["rows"] if r["seed"] == 45)
    assert r45["delta_missing"] == "no same-seed baseline" and "heldout_delta" not in r45
    assert "| symmetric | 45 | 0.500 | no same-seed baseline |" in (doc / "seeds.md").read_text()
    s = {x["variant"]: x for x in data["summary"]}["symmetric"]
    assert s["n_seeds"] == 4 and s["n_deltas_heldout"] == 3
    # the main table is unchanged by the replicates: it still lists only the chain's runs
    main = {r["variant"] for r in json.load(open(doc / "variants.json"))["Toy"]["metrics"]}
    assert main == {"baseline", "symmetric"}
