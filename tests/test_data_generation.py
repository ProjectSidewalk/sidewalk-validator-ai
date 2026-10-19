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
    rows[-2]["pano_y"] = 5000      # was the legacy row; now pano_y > pano_height: also dropped
    rows[-3]["pano_width"] = 0     # non-GSV row stays counted as non_gsv (first failing check)
    assert len(rows) == 30
    path = tmp_path / "toy-city.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def select(labels, **kw):
    eligible, _ = dg.eligible_labels(labels, "CurbRamp")
    return dg.select_training_labels(eligible, **kw)


def test_select_counts_and_balance(tmp_path):
    df = select(fixture_labels(tmp_path))
    assert (df.label_class == "incorrect").sum() == 6
    assert (df.label_class == "correct").sum() == int(0.55 / 0.45 * 6)  # 7: downsampled from 12
    assert set(df.author) == {"human"}
    assert (df.pano_source == "gsv").all() and (df.pano_y >= 0).all()


def test_select_is_deterministic_and_seeded(tmp_path):
    labels = fixture_labels(tmp_path)
    a = select(labels, seed=42)
    b = select(labels, seed=42)
    c = select(labels, seed=1)
    assert a.label_id.tolist() == b.label_id.tolist()
    assert a.label_id.tolist() != c.label_id.tolist()


def test_select_does_not_depend_on_api_row_order(tmp_path):
    labels = fixture_labels(tmp_path)
    shuffled = labels.sample(frac=1, random_state=3).reset_index(drop=True)
    assert select(labels).label_id.tolist() == select(shuffled).label_id.tolist()


def test_downsample_is_stable_when_the_pool_changes():
    """Ranking by sha256(seed:label_uid): one correct label leaving the pool changes at most one kept label."""
    n = 200
    labels = pd.DataFrame({"city": "c", "label_id": range(n), "author": "human",
                           "label_class": ["incorrect" if i < 60 else "correct" for i in range(n)]})
    a = dg.select_training_labels(labels, seed=42)
    kept = set(a.loc[a.label_class == "correct", "label_id"])
    gone = sorted(kept)[0]
    b = dg.select_training_labels(labels[labels.label_id != gone], seed=42)
    kept_b = set(b.loc[b.label_class == "correct", "label_id"])
    assert len(kept) == len(kept_b) == int(0.55 / 0.45 * 60)
    assert kept - kept_b == {gone} and len(kept_b - kept) == 1
    unkept = (set(range(60, n)) - kept).pop()  # removing a label that was not kept changes nothing
    c = dg.select_training_labels(labels[labels.label_id != unkept], seed=42)
    assert set(c.loc[c.label_class == "correct", "label_id"]) == kept


def test_priors_lock_keeps_concurrent_writers(tmp_path):
    import multiprocessing as mp
    path = str(tmp_path / "deployment_priors.json")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_write_priors_many, args=(path, lt)) for lt in ("CurbRamp", "Obstacle")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    with open(path) as f:
        assert set(json.load(f)) == {"CurbRamp", "Obstacle"}


def test_include_ai_labels(tmp_path):
    df = select(fixture_labels(tmp_path), include_ai_labels=True)
    assert (df.label_class == "incorrect").sum() == 8
    assert (df.author == "ai").sum() >= 2  # both AI incorrect rows; AI correct rows compete in the downsample
    assert df.loc[df.author == "ai", "user_id"].eq(AI).all()


def test_expert_votes_add_labels_the_margin_rule_skips(tmp_path):
    labels = fixture_labels(tmp_path)
    expert = lambda vote, other=None: json.dumps(
        [{"user_id": "ex", "validation": vote, "validator_type": "Human"}]
        + ([{"user_id": "c", "validation": other, "validator_type": "Human"}] if other else []))
    extra = labels.iloc[:4].copy()
    extra["label_id"] = [101, 102, 103, 104]
    extra["validations"] = [expert("Disagree"), expert("Agree", "Agree"), expert("Disagree", "Agree"),
                            json.dumps([{"user_id": "ex", "validation": "Disagree", "validator_type": "AI"}])]
    labels = pd.concat([labels, extra], ignore_index=True)

    base, _ = dg.eligible_labels(labels, "CurbRamp")
    assert set(base.class_source) == {"margin"} and not base.label_id.isin([101, 102, 103, 104]).any()

    with_ex, _ = dg.eligible_labels(labels, "CurbRamp", experts=dg.parse_experts("someone=ex"))
    added = with_ex[with_ex.class_source == "expert"].set_index("label_id").label_class.to_dict()
    assert added == {101: "incorrect", 102: "correct"}  # 103 contradicted by a human; 104 is not a Human vote
    # Margin-rule labels are unchanged, and the prior ignores the expert-added ones.
    pd.testing.assert_frame_equal(with_ex[with_ex.class_source == "margin"].reset_index(drop=True),
                                  base.reset_index(drop=True))
    assert (dg.deployment_prior_entry(with_ex, ["toy-city"], 33, measured_on="x")
            == dg.deployment_prior_entry(base, ["toy-city"], 33, measured_on="x"))
    sel = dg.select_training_labels(with_ex)
    assert (sel.label_class == "incorrect").sum() == 7 and 101 in set(sel.label_id)


def test_geometry_bounds_are_dropped_and_counted(tmp_path):
    eligible, dropped = dg.eligible_labels(fixture_labels(tmp_path), "CurbRamp")
    assert dict(dropped) == {"non_gsv": 1, "pano_y_out_of_range": 1}
    bad = pd.DataFrame({"pano_source": ["gsv"] * 4, "pano_x": [1, 1, np.inf, 1], "pano_y": [1, 1, 1, 1],
                        "pano_width": [10, -1, 10, 10], "pano_height": [10, 10, 10, np.nan]})
    keep, why = dg.valid_geometry(bad)
    assert keep.tolist() == [True, False, False, False]
    assert dict(why) == {"bad_pano_size": 2, "pano_x_nonfinite": 1}


def test_deployment_prior_is_pre_downsampling_and_human_authored(tmp_path):
    eligible, _ = dg.eligible_labels(fixture_labels(tmp_path), "CurbRamp")
    e = dg.deployment_prior_entry(eligible, ["toy-city"], 29, measured_on="2026-10-17")
    assert e == {"correct": 12, "incorrect": 6, "measured_on": "2026-10-17",
                 "source": "rawLabels of toy-city, human votes only", "cities": ["toy-city"], "n_rows": 29}


@pytest.mark.parametrize("extra, real", [([], True), (["--limit", "3"], False), (["--cities", "toy-city"], False)])
def test_priors_written_only_by_full_runs(tmp_path, monkeypatch, extra, real):
    fixture_labels(tmp_path)  # writes toy-city.csv
    monkeypatch.setattr(dg, "CITIES", {"toy-city": "toy"})
    dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path),
             "--out", str(tmp_path / "out"), *extra])
    full, part = tmp_path / "out" / "deployment_priors.json", tmp_path / "out" / "deployment_priors.partial.json"
    assert full.exists() == real and part.exists() == (not real)
    entry = json.loads((full if real else part).read_text())["CurbRamp"]
    assert (entry["correct"], entry["incorrect"], entry["cities"], entry["n_rows"]) == (12, 6, ["toy-city"], 29)


def test_no_silent_cpu_fallback(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_cuda_torch(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_cuda_torch)
    with pytest.raises(SystemExit, match="allow-cpu"):
        dg.resolve_device(allow_cpu=False)
    assert dg.resolve_device(allow_cpu=True) is None


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
            "crop_path": crop, "status": status, "pano_path": "/store/c/p/p.jpg"}


def test_manifest_is_resumable(tmp_path):
    base = tmp_path / "CurbRamp"
    (base / "correct").mkdir(parents=True)
    path = str(base / "manifest.csv")
    sel = {f"c:{i}": "correct" for i in range(4)}
    with dg.ManifestWriter(path, str(base), sel) as m:
        for i, status in enumerate(["ok", "ok", "no_pano", "error:ValueError"]):
            crop = f"correct/c_{i}.webp"
            if status == "ok":
                (base / crop).write_bytes(b"x")
            m.write(_row(f"c:{i}", status, crop))
        with pytest.raises(AssertionError):
            m.write(_row("c:0", "ok", "correct/c_0.webp"))
    (base / "correct" / "c_1.webp").unlink()  # an ok row whose crop vanished is redone

    with dg.ManifestWriter(path, str(base), sel) as m:
        assert {u for u in [f"c:{i}" for i in range(4)] if m.is_done(u)} == {"c:0"}
        for i in (1, 2, 3):
            m.write(_row(f"c:{i}", "ok", f"correct/c_{i}.webp"))
    df = pd.read_csv(path)
    assert list(df.columns) == dg.MANIFEST_COLUMNS
    assert df.label_uid.is_unique and len(df) == 4 and set(df.status) == {"ok"}


def test_resume_drops_rows_from_a_stale_selection(tmp_path):
    base = tmp_path / "CurbRamp"
    (base / "correct").mkdir(parents=True)
    path = str(base / "manifest.csv")
    with dg.ManifestWriter(path, str(base), {f"c:{i}": "correct" for i in range(3)}) as m:
        for i in range(3):
            (base / f"correct/c_{i}.webp").write_bytes(b"x")
            m.write(_row(f"c:{i}", "ok", f"correct/c_{i}.webp"))
    # new votes: c:1 flipped to incorrect, c:2 no longer qualifies, c:3 is new
    new = {"c:0": "correct", "c:1": "incorrect", "c:3": "correct"}
    with dg.ManifestWriter(path, str(base), new) as m:
        assert m.done == {"c:0"}
        assert dict(m.stale) == {"class_changed": 1, "no_longer_selected": 1}
    assert pd.read_csv(path).label_uid.tolist() == ["c:0"]
    assert (base / "correct/c_2.webp").exists()  # crops are never deleted


def test_check_resume_detects_selection_and_settings_changes(tmp_path):
    a = pd.DataFrame({"label_uid": ["x:1", "x:2"], "label_class": ["correct", "incorrect"]})
    b = a.assign(label_class=["correct", "correct"])
    st = {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti", "renderer": "torch",
          "include_ai_labels": False}
    run = tmp_path / "generation_run.json"
    assert dg.check_resume(str(run), dg.selection_hash(a), st) == (None, {})  # no previous run
    run.write_text(json.dumps({"selection_sha256": dg.selection_hash(a), "settings": st}))
    assert dg.check_resume(str(run), dg.selection_hash(a), st) == (None, {})
    assert dg.check_resume(str(run), dg.selection_hash(b), st)[0] == dg.selection_hash(a)
    assert dg.check_resume(str(run), dg.selection_hash(a), {**st, "tilt_beta": 0.5, "renderer": "numpy"})[1] == \
        {"tilt_beta": (0.95, 0.5), "renderer": ("torch", "numpy")}


def test_manifest_row_uid_and_crop_path():
    row = pd.Series({"city": "seattle-wa", "label_id": 12345, "pano_id": "abc", "label_class": "correct",
                     "author": "human", "human_agree": 3, "human_disagree": 0, "camera_pitch": np.nan})
    rec = dg.manifest_row(row, "CurbRamp", "correct/seattle-wa_12345.webp", 0.95)
    assert rec["label_uid"] == "seattle-wa:12345" and rec["tilt_beta"] == 0.95 and "hard" not in rec
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
                  "expert1": [None, "Agree"], "expert2": [None, None], "ai": ["Agree", "Disagree"]}).to_csv(src, index=False)
    hard_examples.main([str(src), str(tmp_path / "out.csv"), "--experts", "expert1,expert2"])
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(out.columns) == ["label_uid", "reason"]
    assert out.values.tolist() == [["a:1", "unanimous_crowd"], ["a:2", "expert"]]


def test_hard_examples_detects_role_keyed_expert_columns(tmp_path):
    src = tmp_path / "ai_vs_humans.csv"
    pd.DataFrame({"label_uid": ["a:1"], "kind": ["ok"], "h_agree": [3], "h_dis": [0], "expert1": [None],
                  "expert2": ["Disagree"], "ai": ["Agree"]}).to_csv(src, index=False)
    hard_examples.main([str(src), str(tmp_path / "out.csv")])  # no --experts: expert1, expert2 found
    assert pd.read_csv(tmp_path / "out.csv").values.tolist() == [["a:1", "expert"]]


def test_fine_tune_reads_split_csv_with_stable_label_ids(tmp_path):
    import fine_tune_classifier as ft
    base = tmp_path / "Toy"
    base.mkdir()
    pd.DataFrame({"label_uid": ["a:1", "a:2", "a:3"], "label_class": ["incorrect", "correct", "correct"],
                  "crop_path": ["incorrect/a_1.webp", "correct/a_2.webp", "/abs/a_3.webp"],
                  "split": ["train", "val", "test"]}).to_csv(base / "split.csv", index=False)
    train, val = ft.load_split_frames("Toy", str(tmp_path))
    assert ft.ID2LABEL == {0: "correct", 1: "incorrect"}
    assert train.label.tolist() == [1] and val.label.tolist() == [0]
    assert train.image.iloc[0] == str(base / "incorrect" / "a_1.webp")
    assert ft.load_split_frames("Missing", str(tmp_path)) is None


def _write_priors_many(path, label_type):  # module level, so a spawned process can import it
    for i in range(40):
        dg.update_deployment_priors(path, label_type, {"correct": i, "incorrect": i})
