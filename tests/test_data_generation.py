"""data_generation.py on a CPU: label selection, AI-author handling, deployment priors, the resumable manifest."""
import doctest
import json

import numpy as np
import pandas as pd
import pytest

import data_generation as dg
import hard_examples

AI = dg.AI_USER_ID


def _votes(agree, disagree, ai_agree=0):
    v = ([{"user_id": f"h{i}", "validation": "Agree", "validator_type": "Human"} for i in range(agree)]
         + [{"user_id": f"d{i}", "validation": "Disagree", "validator_type": "Human"} for i in range(disagree)]
         + [{"user_id": AI, "validation": "Agree", "validator_type": "AI"} for _ in range(ai_agree)])
    return json.dumps(v)


def fixture_labels(tmp_path):
    """30 CurbRamp rows written to CSV and read back (as the API export would be): 12 clearly correct, 6 clearly
    incorrect, 4 contested/unvalidated, 3 AI-authored correct, 2 AI-authored incorrect, 1 non-GSV, 1 legacy
    negative pano_y, 1 other label type. AI votes on 5 rows would flip them to correct if counted."""
    rows = []

    def add(n, agree, disagree, user="u", source="gsv", pano_y=1000, label_type="CurbRamp", ai_agree=0):
        for _ in range(n):
            i = len(rows) + 1
            rows.append({"label_id": i, "user_id": user, "pano_id": f"p{i // 2}", "pano_source": source,
                         "label_type": label_type, "validations": _votes(agree, disagree, ai_agree),
                         "pano_x": 100, "pano_y": pano_y, "pano_width": 8192, "pano_height": 4096,
                         "camera_pitch": 1.5, "camera_roll": None, "city": "toy-city"})
    add(12, 4, 0)
    add(6, 0, 3)
    add(2, 1, 1, ai_agree=5)       # contested; AI votes must not make these correct
    add(2, 0, 0)
    add(3, 5, 0, user=AI)
    add(2, 0, 4, user=AI)
    add(1, 5, 0, source="mapillary")
    add(1, 5, 0, pano_y=-5)
    add(1, 5, 0, label_type="Obstacle")
    assert len(rows) == 30
    path = tmp_path / "toy-city.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def test_select_counts_and_balance(tmp_path):
    df = dg.select_training_labels(fixture_labels(tmp_path), "CurbRamp")
    assert (df.label_class == "incorrect").sum() == 6
    assert (df.label_class == "correct").sum() == int(0.55 / 0.45 * 6)  # 7: downsampled from 12
    assert set(df.author) == {"human"}
    assert (df.pano_source == "gsv").all() and (df.pano_y >= 0).all()


def test_select_is_deterministic_and_seeded(tmp_path):
    labels = fixture_labels(tmp_path)
    a = dg.select_training_labels(labels, "CurbRamp", seed=42)
    b = dg.select_training_labels(labels, "CurbRamp", seed=42)
    c = dg.select_training_labels(labels, "CurbRamp", seed=1)
    assert a.label_id.tolist() == b.label_id.tolist()
    assert a.label_id.tolist() != c.label_id.tolist()


def test_include_ai_labels(tmp_path):
    df = dg.select_training_labels(fixture_labels(tmp_path), "CurbRamp", include_ai_labels=True)
    assert (df.label_class == "incorrect").sum() == 8
    assert (df.author == "ai").sum() >= 2  # both AI incorrect rows; AI correct rows compete in the downsample
    assert df.loc[df.author == "ai", "user_id"].eq(AI).all()


def test_deployment_prior_is_pre_downsampling_and_human_authored(tmp_path):
    e = dg.deployment_prior_entry(fixture_labels(tmp_path), "CurbRamp", ["toy-city"], measured_on="2026-10-17")
    assert e == {"correct": 12, "incorrect": 6, "measured_on": "2026-10-17",
                 "source": "rawLabels of toy-city, human votes only"}


def test_deployment_priors_merge(tmp_path):
    path = str(tmp_path / "deployment_priors.json")
    dg.update_deployment_priors(path, "Obstacle", {"correct": 1, "incorrect": 2})
    dg.update_deployment_priors(path, "CurbRamp", {"correct": 3, "incorrect": 4})
    dg.update_deployment_priors(path, "Obstacle", {"correct": 5, "incorrect": 6})
    with open(path) as f:
        assert json.load(f) == {"CurbRamp": {"correct": 3, "incorrect": 4}, "Obstacle": {"correct": 5, "incorrect": 6}}


def _row(uid, status, crop):
    city, label_id = uid.split(":")
    return {"label_uid": uid, "city": city, "label_id": label_id, "pano_id": "p", "label_class": "correct",
            "crop_path": crop, "status": status}


def test_manifest_is_resumable(tmp_path):
    base = tmp_path / "CurbRamp"
    (base / "correct").mkdir(parents=True)
    path = str(base / "manifest.csv")
    with dg.ManifestWriter(path, str(base)) as m:
        for i, status in enumerate(["ok", "ok", "no_pano", "error:ValueError"]):
            crop = f"correct/c_{i}.webp"
            if status == "ok":
                (base / crop).write_bytes(b"x")
            m.write(_row(f"c:{i}", status, crop))
        with pytest.raises(AssertionError):
            m.write(_row("c:0", "ok", "correct/c_0.webp"))
    (base / "correct" / "c_1.webp").unlink()  # an ok row whose crop vanished is redone

    with dg.ManifestWriter(path, str(base)) as m:
        assert {u for u in [f"c:{i}" for i in range(4)] if m.is_done(u)} == {"c:0"}
        for i in (1, 2, 3):
            m.write(_row(f"c:{i}", "ok", f"correct/c_{i}.webp"))
    df = pd.read_csv(path)
    assert list(df.columns) == dg.MANIFEST_COLUMNS
    assert df.label_uid.is_unique and len(df) == 4 and set(df.status) == {"ok"}


def test_manifest_row_uid_and_crop_path():
    row = pd.Series({"city": "seattle-wa", "label_id": 12345, "pano_id": "abc", "label_class": "correct",
                     "author": "human", "human_agree": 3, "human_disagree": 0, "camera_pitch": np.nan})
    rec = dg.manifest_row(row, "CurbRamp", "correct/seattle-wa_12345.webp", True, 0.95)
    assert rec["label_uid"] == "seattle-wa:12345" and rec["hard"] == 1 and rec["tilt_beta"] == 0.95
    assert set(rec) == set(dg.MANIFEST_COLUMNS)


def test_depth_sized_crop_clips_near_features():
    img = np.zeros((2048, 2048, 3), np.uint8)
    crop, half, clipped = dg.depth_sized_crop(img, 2.0)  # 3050 px half-width > 1024
    assert clipped and half == 3050 and crop.shape == (640, 640, 3)
    with pytest.raises(ValueError):
        dg.depth_sized_crop(img, 0.0)


@pytest.mark.parametrize("module", [dg, hard_examples])
def test_doctests(module):
    result = doctest.testmod(module)
    assert result.attempted > 0 and result.failed == 0


def test_hard_examples_writes_only_uid_and_reason(tmp_path):
    src = tmp_path / "ai_vs_humans.csv"
    pd.DataFrame({"label_uid": ["a:1", "a:2"], "kind": ["false_agree", "ok"], "h_agree": [0, 1], "h_dis": [4, 1],
                  "jon": [None, "Agree"], "mikey": [None, None], "ai": ["Agree", "Disagree"]}).to_csv(src, index=False)
    hard_examples.main([str(src), str(tmp_path / "out.csv")])
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(out.columns) == ["label_uid", "reason"]
    assert out.values.tolist() == [["a:1", "unanimous_crowd"], ["a:2", "expert"]]
