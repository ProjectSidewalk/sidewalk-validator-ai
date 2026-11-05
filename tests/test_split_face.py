"""split_data.py --group-col face_id (issue #42): block faces joined with their panoramas."""
import doctest

import pandas as pd
import pytest

import data_generation as dg
import split_data


def face_manifest(n_faces=40, per_face=4, seed=0):
    """``n_faces`` faces of ``per_face`` labels, each label on its own panorama; every 5th face is incorrect, and
    face k's first panorama also sees face k+1 on the other side of the street (one shared panorama)."""
    rows = []
    for f in range(n_faces):
        for j in range(per_face):
            uid = f"c:{f * per_face + j}"
            pano = f"p{f}-{j}" if j else (f"p{f - 1}-shared" if f % 2 else f"p{f}-shared")
            rows.append({c: "" for c in dg.MANIFEST_COLUMNS} | {
                "label_uid": uid, "city": "c" if f < n_faces - 4 else "held", "label_id": uid.split(":")[1],
                "pano_id": pano, "face_id": f"c:{f}:{'left' if f % 2 else 'right'}",
                "label_class": "incorrect" if f % 5 == 0 else "correct", "status": "ok",
                "crop_path": f"x/{uid}.webp"})
    rows.append({c: "" for c in dg.MANIFEST_COLUMNS} | {"label_uid": "c:999", "city": "c", "label_id": "999",
                                                        "pano_id": "lonely", "face_id": "", "label_class": "correct",
                                                        "status": "ok", "crop_path": "x/c:999.webp"})
    return pd.DataFrame(rows, columns=dg.MANIFEST_COLUMNS)


def test_face_groups_keep_faces_and_panoramas_whole():
    m = face_manifest()
    df = split_data.assign_splits(m, seed=1, group_col="face_id", holdout_cities=["held"])
    assert (df[df.face_id != ""].groupby("face_id").split.nunique() == 1).all()
    split_data.check_no_shared_panoramas(df)
    # faces 2k and 2k+1 share a panorama, so they always land together
    by_face = df.set_index("face_id").split.groupby(level=0).first()
    for k in range(0, 36, 2):
        assert by_face[f"c:{k}:right"] == by_face[f"c:{k + 1}:left"]
    assert set(df.loc[df.city == "held", "split"]) == {"test_heldout"}


def test_pano_grouping_splits_faces_and_face_grouping_does_not():
    m = face_manifest()
    pano = split_data.assign_splits(m, seed=1, group_col="pano_id")
    assert (pano[pano.face_id != ""].groupby("face_id").split.nunique() > 1).any()  # the leak the flag closes
    assert pano.equals(split_data.assign_splits(m, seed=1))  # pano_id stays the default


def test_face_split_is_deterministic_and_seeded():
    m = face_manifest()
    a = split_data.assign_splits(m, seed=3, group_col="face_id")
    assert a.equals(split_data.assign_splits(m.sample(frac=1, random_state=2), seed=3, group_col="face_id"))
    assert not a.split.equals(split_data.assign_splits(m, seed=4, group_col="face_id").split)
    with pytest.raises(ValueError):
        split_data.assign_splits(m, group_col="street")


def test_main_with_face_groups_and_without_a_face_column(tmp_path):
    base = tmp_path / "Toy"
    base.mkdir()
    face_manifest().to_csv(base / "manifest.csv", index=False)
    args = ["--label_type", "Toy", "--out", str(tmp_path), "--merge-radius-m", "0", "--group-col", "face_id"]
    assert split_data.main(args) == 0
    df = pd.read_csv(base / "split.csv", dtype=str, keep_default_na=False)
    assert (df[df.face_id != ""].groupby("face_id").split.nunique() == 1).all()
    face_manifest().drop(columns="face_id").to_csv(base / "manifest.csv", index=False)
    assert split_data.main(args) == 2
    assert (base / "split.csv").exists()  # refused before computing: an earlier split is left alone


def test_doctests():
    result = doctest.testmod(split_data)
    assert result.attempted > 0 and result.failed == 0
