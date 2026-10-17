"""split_data.py: panorama-grouped, stratified, seeded, idempotent, and it never touches image files."""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

import data_generation as dg
import split_data


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
                "label_type": "Toy", "label_class": cls, "crop_path": crop, "status": status, "hard": 0})
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
    assert split_data.main(["--label_type", "Toy", "--out", str(tmp_path), *extra]) == 0
    return (tmp_path / "Toy" / "split.csv").read_bytes()


@pytest.fixture
def toy(tmp_path):
    base = tmp_path / "Toy"
    base.mkdir()
    write_manifest(base)
    return tmp_path


def test_grouped_ok_rows_and_target_shares(toy):
    run(toy)
    df = pd.read_csv(toy / "Toy" / "split.csv")
    manifest = pd.read_csv(toy / "Toy" / "manifest.csv")
    assert len(df) == (manifest.status == "ok").sum()
    assert list(df.columns) == dg.MANIFEST_COLUMNS + ["split"]
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
