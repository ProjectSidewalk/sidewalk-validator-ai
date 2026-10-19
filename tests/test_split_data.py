"""split_data.py: panorama-grouped (plus nearby-label merging), stratified, seeded, idempotent, and it never touches
image files."""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

import data_generation as dg
import split_data


M_PER_DEG_LAT = 111_195.0  # metres per degree of latitude on the mean-radius sphere split_data uses


def pano_position(p):
    """(lat, lon) of toy panorama ``p``: panoramas sit ~111 m apart along a meridian, except that each panorama with
    ``p % 9`` in {3, 4, 5} is placed 2 m north of panorama ``p - 3`` (same city, since city = p % 3), so those
    pairs see the same physical feature and must be merged at a 3 m radius."""
    if p % 9 in (3, 4, 5):
        lat, lon = pano_position(p - 3)
        return lat + 2.0 / M_PER_DEG_LAT, lon
    return 47.6 + p * 0.001, -122.3


NEAR_PAIRS = [(p - 3, p) for p in range(50) if p % 9 in (3, 4, 5)]


def write_rawlabels(root, manifest_path, seed=1, drop=()):
    """Write ``<root>/rawlabels/<city>.csv`` (label_id, latitude, longitude, pano_id, label_type) for every manifest
    row: each label within 0.5 m of its panorama's position. ``drop`` lists label_uids to leave out."""
    rng = np.random.default_rng(seed)
    m = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    rows = []
    for uid, city, lid, pano in zip(m.label_uid, m.city, m.label_id, m.pano_id):
        lat, lon = pano_position(int(pano[4:]))
        dy, dx = rng.uniform(-0.35, 0.35, 2) / M_PER_DEG_LAT
        if uid not in drop:
            rows.append({"city": city, "label_id": int(lid), "latitude": lat + dy, "longitude": lon + dx,
                         "pano_id": pano, "label_type": "Toy"})
    out = root / "rawlabels"
    out.mkdir(exist_ok=True)
    raw = pd.DataFrame(rows)
    for city, part in raw.groupby("city"):
        part.drop(columns="city").to_csv(out / f"{city}.csv", index=False)
    return out


def write_manifest(base, n_panos=50, seed=0):
    """50 panoramas x 1-4 labels across three cities, with a dummy crop per ok row; returns the manifest path."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_panos):
        city = ["alpha-wa", "beta-or", "gamma"][p % 3]
        for _ in range(rng.integers(1, 5)):
            i = len(rows)
            cls = "incorrect" if rng.random() < 0.45 else "correct"
            crop = f"{cls}/{city}_{i}.webp"
            status = "ok" if i % 17 else "no_pano"
            rows.append({c: "" for c in dg.MANIFEST_COLUMNS} | {
                "label_uid": f"{city}:{i}", "city": city, "label_id": i, "pano_id": f"pano{p:03d}",
                "label_type": "Toy", "label_class": cls, "crop_path": crop, "status": status})
            if status == "ok":
                (base / cls).mkdir(parents=True, exist_ok=True)
                (base / crop).write_bytes(os.urandom(16))
    path = base / "manifest.csv"
    pd.DataFrame(rows, columns=dg.MANIFEST_COLUMNS).to_csv(path, index=False)
    return path


def snapshot(base):
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in base.rglob("*.webp")}


def run(tmp_path, *extra):
    """Run split_data.py on the toy dataset (with its rawlabels/ for coordinates) and return split.csv's bytes."""
    args = ["--label_type", "Toy", "--out", str(tmp_path), "--rawlabels", str(tmp_path / "rawlabels"), *extra]
    assert split_data.main(args) == 0
    return (tmp_path / "Toy" / "split.csv").read_bytes()


@pytest.fixture
def toy(tmp_path):
    base = tmp_path / "Toy"
    base.mkdir()
    write_rawlabels(tmp_path, write_manifest(base))
    return tmp_path


def test_grouped_ok_rows_and_target_shares(toy):
    run(toy)
    df = pd.read_csv(toy / "Toy" / "split.csv")
    manifest = pd.read_csv(toy / "Toy" / "manifest.csv")
    assert len(df) == (manifest.status == "ok").sum()
    cols = list(dg.MANIFEST_COLUMNS)
    cols.insert(cols.index("crop_path"), "hard")
    assert list(df.columns) == cols + ["split"]  # unchanged split.csv layout; hard joined here
    assert (df.hard == 0).all()
    assert set(df.split) == {"train", "val", "test"}
    assert (df.groupby("pano_id").split.nunique() == 1).all()
    share = df.split.value_counts(normalize=True)
    assert 0.6 < share["train"] < 0.8 and 0.07 < share["val"] < 0.25 and 0.07 < share["test"] < 0.25


def test_same_seed_is_byte_identical_and_other_seed_differs(toy):
    a = run(toy)
    b = run(toy)
    c = run(toy, "--seed", "7")
    assert a == b
    assert a != c


def test_holdout_cities_route_whole_cities(toy):
    run(toy, "--holdout-cities", "gamma")
    df = pd.read_csv(toy / "Toy" / "split.csv")
    assert (df[df.city == "gamma"].split == "test_heldout").all()
    assert (df[df.city != "gamma"].split != "test_heldout").all()
    split_data.check_no_shared_panoramas(df)


def test_never_touches_files(toy):
    before = snapshot(toy / "Toy")
    run(toy)
    run(toy, "--holdout-cities", "beta-or")
    assert snapshot(toy / "Toy") == before
    assert sorted(os.listdir(toy / "Toy")) == ["correct", "incorrect", "manifest.csv", "split.csv"]


def test_hard_examples_joined_by_label_uid(toy, tmp_path):
    first = pd.read_csv(toy / "Toy" / "manifest.csv").query("status == 'ok'").label_uid.iloc[:2].tolist()
    hard = tmp_path / "hard.csv"
    pd.DataFrame({"label_uid": first + ["nowhere:1"], "reason": "expert"}).to_csv(hard, index=False)
    run(toy, "--hard-examples", str(hard))
    df = pd.read_csv(toy / "Toy" / "split.csv")
    assert sorted(df.loc[df.hard == 1, "label_uid"]) == sorted(first)
    import evaluate
    import os as _os
    cwd = _os.getcwd()
    try:
        _os.chdir(toy)
        assert evaluate.load_split("Toy").hard.sum() == 2  # evaluate.py reads hard from split.csv
    finally:
        _os.chdir(cwd)


def test_refuses_without_manifest(tmp_path, capsys):
    (tmp_path / "Toy" / "correct").mkdir(parents=True)
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path)]) == 2
    assert "manifest" in capsys.readouterr().err
    assert not (tmp_path / "Toy" / "split.csv").exists()


def test_shared_panorama_detected():
    df = pd.DataFrame({"pano_id": ["a", "a"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        split_data.check_no_shared_panoramas(df)


def test_holdout_pano_shared_with_another_city_goes_whole():
    m = pd.DataFrame({"label_uid": ["x:1", "y:2", "y:3"], "city": ["x", "y", "y"], "pano_id": ["p", "p", "q"],
                      "label_class": ["correct"] * 3, "status": ["ok"] * 3})
    df = split_data.assign_splits(m, holdout_cities=["x"])
    assert df.set_index("label_uid").split.to_dict()["y:2"] == "test_heldout"
    split_data.check_no_shared_panoramas(df)


# --- merging nearby labels (issue 27) -----------------------------------------------------------------------------

# sha256 of split.csv written by split_data.py before merging existed (master 71f661a) on the toy manifest.
PANO_ONLY_SHA = {(): "7d4c2c308463a3506bb728af99463c25cf0299fe5446df665f0b083e877f4a1b",
                 ("--holdout-cities", "gamma"): "13430331a4b45750b5b26c45e860ec3ecea4d1edc83220dfe20a81f9f17c7705"}


def toy_coords(toy):
    m = pd.read_csv(toy / "Toy" / "manifest.csv", dtype=str, keep_default_na=False)
    return split_data.load_coordinates(m, toy / "rawlabels")


def test_merge_puts_near_labels_on_one_side(toy, capsys):
    run(toy)
    out = capsys.readouterr().out
    df = pd.read_csv(toy / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    per_pano = df.groupby("pano_id").split.first()
    pairs = [(f"pano{a:03d}", f"pano{b:03d}") for a, b in NEAR_PAIRS]
    pairs = [(a, b) for a, b in pairs if a in per_pano and b in per_pano]
    assert len(pairs) >= 10
    assert all(per_pano[a] == per_pano[b] for a, b in pairs)
    split_data.check_no_shared_panoramas(df)
    rep = split_data.proximity_report(df, toy_coords(toy), (3.0, 5.0)).set_index("split")
    assert (rep["within_3m"] == 0).all() and (rep["within_5m"] == 0).all()  # toy panoramas are ~111 m apart
    assert f"{len(pairs)} groups span more than one panorama" in out
    assert "within_2m" in out and "within_3m" in out and "within_10m" in out


@pytest.mark.parametrize("extra", list(PANO_ONLY_SHA))
def test_radius_zero_reproduces_panorama_only_split(toy, extra):
    with_coords = run(toy, "--merge-radius-m", "0", *extra)
    assert hashlib.sha256(with_coords).hexdigest() == PANO_ONLY_SHA[extra]
    # and coordinates are not needed at radius 0
    (toy / "rawlabels").rename(toy / "elsewhere")
    assert split_data.main(["--label_type", "Toy", "--out", str(toy), "--merge-radius-m", "0", *extra]) == 0
    assert (toy / "Toy" / "split.csv").read_bytes() == with_coords


def test_merge_keeps_split_csv_layout(toy):
    run(toy)
    df = pd.read_csv(toy / "Toy" / "split.csv")
    cols = list(dg.MANIFEST_COLUMNS)
    cols.insert(cols.index("crop_path"), "hard")
    assert list(df.columns) == cols + ["split"]


def test_proximity_report_hand_made():
    north = lambda m: 47.0 + m / M_PER_DEG_LAT
    d = pd.DataFrame({"label_uid": ["t", "a", "b", "c", "d", "v", "h"],
                      "split": ["train", "test", "test", "test", "test", "val", "test_heldout"]})
    c = pd.DataFrame({"latitude": [north(0), north(1), north(3), north(7), north(20), north(4), north(50)],
                      "longitude": [-122.0] * 7}, index=d.label_uid)
    rep = split_data.proximity_report(d, c).set_index("split")
    assert rep.loc["test"].to_dict() == {"labels": 4, "within_2m": 0.25, "within_5m": 0.5, "within_10m": 0.75}
    assert rep.loc["val"].to_dict() == {"labels": 1, "within_2m": 0.0, "within_5m": 1.0, "within_10m": 1.0}
    assert rep.loc["test_heldout", "within_10m"] == 0.0
    assert "train" not in rep.index


def test_merge_is_transitive_and_counts():
    north = lambda m: 47.0 + m / M_PER_DEG_LAT
    g = pd.Series(["p3", "p1", "p2", "p4", "p4"])  # p1-p2-p3 chain 2.5 m apart; p4 far away
    merged, st = split_data.merge_nearby_groups(g, [north(5), north(0), north(2.5), north(100), north(101)],
                                                [-122.0] * 5, 3.0)
    assert merged.tolist() == ["p1", "p1", "p1", "p4", "p4"]
    assert st == {"groups": 4, "merged_groups": 2, "groups_merged": 2, "multi_panorama_groups": 1,
                  "max_original_groups": 3, "max_labels": 3}


@pytest.mark.parametrize("case", ["no_source", "missing_city"])
def test_no_coordinate_source_fails_loudly(toy, capsys, case):
    args = ["--label_type", "Toy", "--out", str(toy)]
    if case == "no_source":
        (toy / "rawlabels").rename(toy / "elsewhere")  # not at the <out>/rawlabels default either
    else:
        (toy / "rawlabels" / "beta-or.csv").unlink()
    assert split_data.main(args) == 2
    err = capsys.readouterr().err
    assert "needs label coordinates" in err and "--merge-radius-m 0" in err
    assert case == "no_source" or "beta-or" in err
    assert not (toy / "Toy" / "split.csv").exists()


def test_rawlabels_defaults_to_out_rawlabels(toy):
    assert split_data.main(["--label_type", "Toy", "--out", str(toy)]) == 0
    assert (toy / "Toy" / "split.csv").read_bytes() == run(toy)


@pytest.mark.parametrize("case", ["missing_label", "nan_coordinate"])
def test_a_few_missing_coordinates_are_tolerated_and_counted(toy, capsys, case):
    manifest = toy / "Toy" / "manifest.csv"
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    ok_uid = m.loc[m.status == "ok", "label_uid"].iloc[0]
    if case == "missing_label":
        write_rawlabels(toy, manifest, drop={ok_uid})
    else:
        city, lid = ok_uid.split(":")
        path = toy / "rawlabels" / f"{city}.csv"
        raw = pd.read_csv(path)
        raw.loc[raw.label_id == int(lid), "latitude"] = np.nan
        raw.to_csv(path, index=False)
    run(toy)  # 1 of 119 ok labels (0.8%) is under the 1% default
    out = capsys.readouterr().out
    assert "1 ok labels have no coordinates" in out
    df = pd.read_csv(toy / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    rep = split_data.proximity_report(df, toy_coords(toy)).set_index("split")
    assert rep["labels"].sum() == (df.split != "train").sum() - (df.set_index("label_uid").split[ok_uid] != "train")
    capsys.readouterr()
    args = ["--label_type", "Toy", "--out", str(toy), "--max-missing-coords-frac", "0"]
    assert split_data.main(args) == 2
    err = capsys.readouterr().err
    assert "--max-missing-coords-frac" in err and ok_uid in err
    assert "Removed the stale" in err and not (toy / "Toy" / "split.csv").exists()


def test_coordinates_join_on_city_and_label_id(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"label_id": [7], "latitude": [10.0], "longitude": [20.0]}).to_csv(raw / "x.csv", index=False)
    pd.DataFrame({"label_id": [7, 8], "latitude": [30.0, 31.0], "longitude": [40.0, 41.0]}).to_csv(
        raw / "y.csv", index=False)
    m = pd.DataFrame({"label_uid": ["x:7", "y:7"], "city": ["x", "y"], "label_id": ["7", "7"], "status": ["ok", "ok"]})
    c = split_data.load_coordinates(m, raw)
    assert c.loc["x:7"].tolist() == [10.0, 20.0] and c.loc["y:7"].tolist() == [30.0, 40.0]


def test_manifest_coordinates_used_without_rawlabels():
    m = pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "city": ["a"] * 3, "pano_id": ["p", "q", "r"],
                      "label_class": ["correct"] * 3, "status": ["ok"] * 3,
                      "latitude": ["47.0", str(47.0 + 1 / M_PER_DEG_LAT), "48.0"], "longitude": ["-122.0"] * 3})
    df, st = split_data.assign_splits(m, merge_radius_m=3, coords=split_data.load_coordinates(m),
                                      return_stats=True)
    s = df.set_index("label_uid").split
    assert s["a:1"] == s["a:2"] and st["groups_merged"] == 1


def test_merged_group_touching_holdout_city_goes_whole():
    north = lambda m: 47.0 + m / M_PER_DEG_LAT
    m = pd.DataFrame({"label_uid": ["x:1", "y:1", "y:2"], "city": ["x", "y", "y"], "pano_id": ["p", "q", "r"],
                      "label_class": ["correct"] * 3, "status": ["ok"] * 3})
    c = pd.DataFrame({"latitude": [north(0), north(1.5), north(500)], "longitude": [-122.0] * 3},
                     index=m.label_uid)
    s = split_data.assign_splits(m, holdout_cities=["x"], merge_radius_m=3, coords=c).set_index("label_uid").split
    assert s["x:1"] == "test_heldout" and s["y:1"] == "test_heldout"  # y:1 is x:1's feature, seen from pano q
    assert s["y:2"] != "test_heldout"
    s0 = split_data.assign_splits(m, holdout_cities=["x"], merge_radius_m=0).set_index("label_uid").split
    assert s0["y:1"] != "test_heldout"


def test_merge_is_deterministic_and_row_order_free(toy):
    a = run(toy)
    b = run(toy)
    manifest = toy / "Toy" / "manifest.csv"
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    m.sample(frac=1, random_state=3).to_csv(manifest, index=False)  # same rows, shuffled
    c = run(toy)
    assert a == b == c
    assert a != run(toy, "--merge-radius-m", "0")


def test_merge_radius_is_pinned():
    north = lambda m: 47.0 + np.degrees(m / split_data.EARTH_RADIUS_M)  # exact metres along a meridian
    g = pd.Series(["a", "b"])
    merge = lambda d, r=3.0: split_data.merge_nearby_groups(g, [north(0), north(d)], [-122.0] * 2, r)[1]["groups_merged"]
    assert merge(2.5) == 1
    assert merge(3.5) == 0
    # The contract is haversine_m <= radius (the proximity report relies on it, so within_<radius>m is exactly 0):
    # a pair exactly at the radius merges, and one a hair beyond it does not.
    one = lambda v: np.array([v], dtype=float)  # arrays, as inside the merge (scalar maths can differ by an ulp)
    d = float(split_data.haversine_m(one(north(0)), one(-122.0), one(north(3.0)), one(-122.0))[0])
    assert abs(d - 3.0) < 1e-6
    assert merge(3.0, r=d) == 1
    assert merge(3.0, r=float(np.nextafter(d, 0))) == 0
    c = pd.DataFrame({"latitude": [north(0), north(3.0)], "longitude": [-122.0] * 2}, index=["t", "v"])
    rep = split_data.proximity_report(pd.DataFrame({"label_uid": ["t", "v"], "split": ["train", "test"]}), c, (d,))
    assert rep.iloc[0, 2] == 1.0


def test_labels_without_coordinates_are_never_merged_by_distance():
    g = pd.Series(["a", "b", "b", "c"])
    merged, st = split_data.merge_nearby_groups(g, [47.0, np.nan, 47.1, 47.0], [-122.0, -122.0, -122.0, np.nan], 3.0)
    assert merged.tolist() == ["a", "b", "b", "c"] and st["groups_merged"] == 0


def write_chain(root, n=1000, chain=300, spacing_m=2.0):
    """The review's chain: ``n`` labels, each on its own panorama in one city, the first ``chain`` of them 2 m apart
    in a line (one merged group at 3 m), the rest ~111 m apart. Writes manifest.csv and rawlabels/; no crops."""
    base = root / "Toy"
    base.mkdir()
    rows, raw = [], []
    for i in range(n):
        cls = "incorrect" if i % 2 else "correct"
        rows.append({c: "" for c in dg.MANIFEST_COLUMNS} | {
            "label_uid": f"c:{i}", "city": "c", "label_id": i, "pano_id": f"p{i:04d}", "label_type": "Toy",
            "label_class": cls, "crop_path": f"{cls}/c_{i}.webp", "status": "ok"})
        lat = 47.0 + (i * spacing_m if i < chain else 10_000 + i * 111.0) / M_PER_DEG_LAT
        raw.append({"label_id": i, "latitude": lat, "longitude": -122.0})
    pd.DataFrame(rows, columns=dg.MANIFEST_COLUMNS).to_csv(base / "manifest.csv", index=False)
    (root / "rawlabels").mkdir()
    pd.DataFrame(raw).to_csv(root / "rawlabels" / "c.csv", index=False)


def test_giant_chained_group_stops_the_split(tmp_path, capsys):
    write_chain(tmp_path)
    (tmp_path / "Toy" / "split.csv").write_text("stale\n")
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "300 of 1000 train/val/test rows" in err and "--max-group-share" in err and "0 = no limit" in err
    assert "Removed the stale" in err and not (tmp_path / "Toy" / "split.csv").exists()


def test_giant_chained_group_warns_when_allowed(tmp_path, capsys):
    write_chain(tmp_path)
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path), "--max-group-share", "0"]) == 0
    err = capsys.readouterr().err
    assert "Warning: the largest group holds 300 rows (30.0% of 1000 train/val/test rows)" in err
    df = pd.read_csv(tmp_path / "Toy" / "split.csv")
    share = df.split.value_counts(normalize=True)
    assert f"{share['train']:.3f} / {share['val']:.3f} / {share['test']:.3f}" in err
    assert share["train"] > 0.73  # the chain cannot be balanced, which is what the warning is for


def test_holdout_pull_in_is_reported(tmp_path, capsys):
    base = tmp_path / "Toy"
    base.mkdir()
    north = lambda m: 47.0 + m / M_PER_DEG_LAT
    rows, raw = [], {"x": [], "y": []}
    # y on p shares x's panorama; y on q is 1.5 m from x and y on s 2 m beyond it (a chain): 2 by merge, 1 by panorama
    layout = [("x", "p", 0), ("y", "q", 1.5), ("y", "s", 3.5), ("y", "p", 0.5), ("y", "r", 900)]
    for i, (city, pano, metres) in enumerate(layout):
        rows.append({c: "" for c in dg.MANIFEST_COLUMNS} | {
            "label_uid": f"{city}:{i}", "city": city, "label_id": i, "pano_id": pano, "label_type": "Toy",
            "label_class": "correct", "crop_path": f"correct/{city}_{i}.webp", "status": "ok"})
        raw[city].append({"label_id": i, "latitude": north(metres), "longitude": -122.0})
    pd.DataFrame(rows, columns=dg.MANIFEST_COLUMNS).to_csv(base / "manifest.csv", index=False)
    (tmp_path / "rawlabels").mkdir()
    for city, r in raw.items():
        pd.DataFrame(r).to_csv(tmp_path / "rawlabels" / f"{city}.csv", index=False)
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path), "--holdout-cities", "x",
                            "--max-group-share", "0"]) == 0
    assert "Rows of other cities pulled into test_heldout: 3 (2 by a merged group, the rest by a shared panorama)" \
        in capsys.readouterr().out


def test_export_ids_are_stripped_and_unrelated_rows_ignored(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # " 7" is label 7; label 9 has conflicting duplicates but the manifest does not need it
    (raw / "x.csv").write_text("label_id,latitude,longitude\n 7,10.0,20.0\n9,1.0,1.0\n9,2.0,2.0\n")
    m = pd.DataFrame({"label_uid": ["x:7"], "city": ["x"], "label_id": ["7"], "status": ["ok"]})
    assert split_data.load_coordinates(m, raw).loc["x:7"].tolist() == [10.0, 20.0]
    (raw / "x.csv").write_text("label_id,latitude,longitude\n 7,10.0,20.0\n7,11.0,20.0\n")
    with pytest.raises(ValueError, match="x:7 with conflicting"):
        split_data.load_coordinates(m, raw)


def test_radius_zero_never_guards_small_manifests(tmp_path, capsys):
    """The README smoke test (--limit 20) and any tiny manifest must behave as before merging at radius 0, and the
    absolute floor keeps the guard off small manifests at the default radius too."""
    base = tmp_path / "Toy"
    base.mkdir()
    path = write_manifest(base)
    pd.read_csv(path, dtype=str, keep_default_na=False).iloc[:30].to_csv(path, index=False)
    write_rawlabels(tmp_path, path)
    (base / "split.csv").write_text("old\n")
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path), "--merge-radius-m", "0"]) == 0
    assert "Error" not in capsys.readouterr().err
    assert (base / "split.csv").read_text() != "old\n"
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path)]) == 0


def test_guard_counts_only_train_val_test_rows(tmp_path):
    """A big group in a held-out city does not trip --max-group-share (it never goes through the filler)."""
    write_chain(tmp_path)
    m = pd.read_csv(tmp_path / "Toy" / "manifest.csv", dtype=str, keep_default_na=False)
    m.loc[:299, "city"] = "h"
    m.loc[:299, "label_uid"] = "h:" + m.loc[:299, "label_id"]
    m.to_csv(tmp_path / "Toy" / "manifest.csv", index=False)
    raw = pd.read_csv(tmp_path / "rawlabels" / "c.csv")
    raw.iloc[:300].to_csv(tmp_path / "rawlabels" / "h.csv", index=False)
    raw.iloc[300:].to_csv(tmp_path / "rawlabels" / "c.csv", index=False)
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path), "--holdout-cities", "h"]) == 0
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path)]) == 2


@pytest.mark.parametrize("args", [["--merge-radius-m", "-1"], ["--hard-examples", "nowhere.csv"],
                                  ["--manifest", "nowhere.csv"]])
def test_errors_before_loading_leave_split_csv_alone(toy, args):
    (toy / "Toy" / "split.csv").write_text("previous\n")
    assert split_data.main(["--label_type", "Toy", "--out", str(toy), *args]) == 2
    assert (toy / "Toy" / "split.csv").read_text() == "previous\n"


def test_exception_while_splitting_removes_split_csv(toy, monkeypatch):
    (toy / "Toy" / "split.csv").write_text("previous\n")
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(split_data, "check_no_shared_panoramas", boom)
    with pytest.raises(RuntimeError):
        split_data.main(["--label_type", "Toy", "--out", str(toy)])
    assert not (toy / "Toy" / "split.csv").exists() and not (toy / "Toy" / "split.csv.tmp").exists()


def test_radius_zero_with_missing_coordinates_succeeds(toy, capsys):
    manifest = toy / "Toy" / "manifest.csv"
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    gone = set(m.loc[m.status == "ok", "label_uid"].iloc[:5])  # 4% of ok labels, over the 1% merge limit
    write_rawlabels(toy, manifest, drop=gone)
    run(toy, "--merge-radius-m", "0")
    out = capsys.readouterr().out
    assert "5 ok labels have no coordinates" in out
    df = pd.read_csv(toy / "Toy" / "split.csv", dtype=str, keep_default_na=False)
    # the printed report's labels column counts only labels with coordinates
    report = out.split("within r metres:\n")[1].splitlines()
    printed = {line.split()[0]: int(line.split()[1]) for line in report[1:]}
    for s in ("val", "test"):
        assert printed[s] == ((df.split == s) & ~df.label_uid.isin(gone)).sum()
    (toy / "rawlabels").rename(toy / "elsewhere")  # no source at all: a note, still a split
    assert split_data.main(["--label_type", "Toy", "--out", str(toy), "--merge-radius-m", "0"]) == 0
    assert "Note: no proximity report" in capsys.readouterr().err
