"""evaluate.load_split: the split.csv manifest, and the image-folder fallback."""
import pandas as pd
import pytest

import evaluate


def test_split_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Toy" / "crops").mkdir(parents=True)
    (tmp_path / "Toy" / "crops" / "a_1.webp").write_bytes(b"")
    pd.DataFrame([{"label_uid": "a:1", "city": "a", "label_id": 1, "pano_id": "p", "label_type": "Toy",
                   "label_class": "correct", "crop_path": "crops/a_1.webp", "split": "test", "hard": 0}]
                 ).to_csv(tmp_path / "Toy" / "split.csv", index=False)
    df = evaluate.load_split("Toy")
    assert df.crop_path.iloc[0] == "Toy/crops/a_1.webp"


def test_split_csv_missing_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Toy").mkdir()
    pd.DataFrame([{"label_uid": "a:1", "split": "test"}]).to_csv(tmp_path / "Toy" / "split.csv", index=False)
    with pytest.raises(ValueError):
        evaluate.load_split("Toy")


def test_imagefolder_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for split, cls, name in [("val", "correct", "seattle_12.webp"), ("test", "incorrect", "new_taipei_7.webp")]:
        (tmp_path / "Toy" / "ds" / split / cls).mkdir(parents=True)
        (tmp_path / "Toy" / "ds" / split / cls / name).write_bytes(b"")
    with pytest.warns(UserWarning):
        df = evaluate.load_split("Toy")
    assert sorted(df.label_uid) == ["new_taipei:7", "seattle:12"]
    assert set(df.split) == {"val", "test"}
