"""Tests for analysis/crop_constant_sweep.py and its two klone launchers (issue #21). CPU only, synthetic data, no
network: a fake panorama reader and renderer stand in for the store and the torch render, so nothing here needs a
GPU, torch or the panorama store."""
import doctest
import json
import os
import shutil
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import crop_pose
import data_generation as dg
from analysis import crop_constant_sweep as cc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ("CurbRamp", "SurfaceProblem")
SPLIT_COLS = ["label_uid", "city", "label_id", "pano_id", "label_type", "label_class", "class_source", "author",
              "human_agree", "human_disagree", "severity", "tags", "high_quality_user", "image_capture_date",
              "pano_source", "pano_path", "pano_native_w", "pano_native_h", "camera_pitch", "camera_roll",
              "tilt_applied", "tilt_beta", "depth_m", "crop_half_px", "clipped", "hard", "crop_path", "status", "split"]
NATIVE = (16384, 8192)


def test_doctests():
    res = doctest.testmod(cc)
    assert res.failed == 0 and res.attempted > 0


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic round 3: split files, rawLabels, generation records, stored crops; a fake reader and renderer
# ---------------------------------------------------------------------------------------------------------------------

def fake_view(x_norm, y_norm):
    """A deterministic 2048 x 2048 view that differs by label, smooth enough for WebP."""
    i = np.arange(dg.VIEW_SIZE, dtype=np.int64)
    base = (np.add.outer(i, i) // 16 + int(x_norm * 997) + int(y_norm * 331)) % 256
    return np.stack([base, base[::-1], base.T], -1).astype(np.uint8)


class FakeRender:
    def __init__(self):
        self.calls = []

    def __call__(self, pano_rgb, x_norm, y_norm, camera_pitch=None, camera_roll=None, apply_tilt=True, beta=0.95,
                 device=None):
        self.calls.append((x_norm, y_norm, camera_pitch, camera_roll, apply_tilt, beta, device))
        return fake_view(x_norm, y_norm), apply_tilt


class FakeReader:
    """``read_store_panorama``'s contract: ``no_pano`` for a pano id starting MISS, a native size of 8192 x 4096
    (not the recorded 16384 x 8192) for one starting SMALL."""

    def __init__(self):
        self.calls = []

    def __call__(self, store, city, pano_id):
        self.calls.append((store, city, pano_id))
        path = dg.pano_store_path(store, city, pano_id)
        if pano_id.startswith("MISS"):
            return "no_pano", {"pano_path": None, "pano_native_w": None, "pano_native_h": None}
        w, h = (8192, 4096) if pano_id.startswith("SMALL") else NATIVE
        return np.zeros((4, 8, 3), np.uint8), {"pano_path": path, "pano_native_w": w, "pano_native_h": h}


def make_round3(root, panos=None, labels_per_pano=3, write_stored=True):
    """``<root>/<Type>/split.csv``, ``generation_run.json``, the stored round-3 crops, and ``<root>/rawlabels``.

    ``panos`` maps a type to its ``(city, pano_id)`` list. The first label's recorded ``crop_half_px`` is one more
    than ``int(6100 / depth_m)`` (round 3 has 17 such rows), so a re-cut that recomputes 6100 instead of reading the
    record would show."""
    panos = panos or {"CurbRamp": [("pub-a", "AApano1"), ("pub-a", "ABpano2"), ("secret-town", "BBpano3"),
                                   ("secret-town", "MISSpano4")],
                      "SurfaceProblem": [("pub-a", "CCpano5"), ("secret-town", "SMALLpano6"),
                                         ("pub-a", "DDpano7")]}
    raw = {}
    n = 0
    splits = ["train", "val", "test", "test_heldout"]
    for t, plist in panos.items():
        rows = []
        for city, pano in plist:
            for j in range(labels_per_pano):
                n += 1
                lid = str(1000 + n)
                depth = round(6.0 + (n * 1.37) % 20, 4)
                half = int(6100 / depth) + (1 if n == 1 else 0)
                x, y = 1000 + 400 * j + n, 4300 + 37 * n
                rows.append({
                    "label_uid": f"{city}:{lid}", "city": city, "label_id": lid, "pano_id": pano, "label_type": t,
                    "label_class": "incorrect" if n % 2 else "correct", "class_source": "margin", "author": "human",
                    "human_agree": 3, "human_disagree": 0, "severity": 1 + n % 3, "tags": '["narrow"]',
                    "high_quality_user": True, "image_capture_date": "2019-05", "pano_source": "gsv",
                    "pano_path": f"/stores/r3/{t}/{city}/{pano[:2]}/{pano}.jpg", "pano_native_w": NATIVE[0],
                    "pano_native_h": NATIVE[1], "camera_pitch": 1.25, "camera_roll": "", "tilt_applied": 1,
                    "tilt_beta": 0.95, "depth_m": depth, "crop_half_px": half, "clipped": 0, "hard": 0,
                    "crop_path": f"{'incorrect' if n % 2 else 'correct'}/{city}_{lid}.webp", "status": "ok",
                    "split": splits[n % 4]})
                raw.setdefault(city, []).append({"label_id": lid, "pano_id": pano, "pano_x": x, "pano_y": y,
                                                 "pano_width": NATIVE[0], "pano_height": NATIVE[1]})
        df = pd.DataFrame(rows, columns=SPLIT_COLS)
        os.makedirs(os.path.join(root, t), exist_ok=True)
        df.to_csv(os.path.join(root, t, "split.csv"), index=False)
        with open(os.path.join(root, t, "generation_run.json"), "w") as f:
            json.dump({"label_type": t, "tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                       "renderer": "torch", "package_source": "pkg@abc",
                       "settings": {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                                    "renderer": "torch", "package_source": "pkg@abc"}, "seed": 42}, f)
    os.makedirs(os.path.join(root, "rawlabels"), exist_ok=True)
    for city, rr in raw.items():
        pd.DataFrame(rr).to_csv(os.path.join(root, "rawlabels", f"{city}.csv"), index=False)
    with open(os.path.join(root, "deployment_priors.json"), "w") as f:
        json.dump({"CurbRamp": {}}, f)
    frame = cc.load_frame(root, os.path.join(root, "rawlabels"), tuple(panos))
    if write_stored:
        for r in frame.to_dict("records"):
            view = fake_view(r["pano_x"] / r["pano_width"], r["pano_y"] / r["pano_height"])
            crop, _, _ = dg.centre_crop(view, int(r["crop_half_px"]))
            dg.save_webp(crop, os.path.join(root, r["label_type"], r["crop_path"]))
    return frame


@pytest.fixture
def r3(tmp_path):
    root = str(tmp_path / "round3")
    frame = make_round3(root)
    return root, frame


def run_recrop(root, frame, work, max_labels=1000, **kw):
    render, reader = kw.pop("render", FakeRender()), kw.pop("pano_reader", FakeReader())
    plan = cc.chunk_plan(frame, max_labels)
    for k in sorted(plan.unique()):
        cc.recrop(frame, work, k, max_labels, pano_store=kw.get("pano_store"), stored_root=root, device=None,
                  pano_reader=reader, render=render)
    return render, reader


# ---------------------------------------------------------------------------------------------------------------------
# Chunks and the stage list
# ---------------------------------------------------------------------------------------------------------------------

def test_chunk_plan_keeps_a_panorama_whole_and_respects_the_cap(r3):
    _, frame = r3
    plan = cc.chunk_plan(frame, 7)
    assert plan.index.equals(frame.index)
    per_pano = frame.assign(c=plan).groupby(["city", "pano_id"]).c.nunique()
    assert (per_pano == 1).all()                                     # one read per panorama
    assert frame.assign(c=plan).groupby("c").size().max() <= 7
    assert cc.chunk_plan(frame.sample(frac=1, random_state=3), 7).sort_index().equals(plan.sort_index())
    with pytest.raises(ValueError, match="more labels than"):
        cc.chunk_plan(frame, 2)                                      # a panorama holds 3 labels


def test_chunk_size_fits_the_time_limit_at_the_slow_rate():
    """chunk labels x 3.5 s x 1.5 must fit inside --time, so a slow node does not time out in a requeue loop."""
    assert cc.parse_time("8:00:00") == 28800 and cc.parse_time("1-02:00:00") == 93600
    assert cc.parse_time("30:00") == 1800
    cap = cc.max_labels_for("8:00:00")
    assert cap * cc.SLOW_SECONDS_PER_LABEL * cc.TIME_MARGIN <= 28800
    assert (cap + 1) * cc.SLOW_SECONDS_PER_LABEL * cc.TIME_MARGIN > 28800
    assert cc.SLOW_SECONDS_PER_LABEL == 3.5 and cc.TIME_MARGIN == 1.5


def test_stage_list_is_deduplicated_store_relative(r3, tmp_path):
    root, frame = r3
    files = cc.stage_files(frame)
    assert files == sorted(set(files))
    assert len(files) == frame[["city", "pano_id"]].drop_duplicates().shape[0]
    assert "pub-a/AA/AApano1.jpg" in files and all(not f.startswith("/") for f in files)
    store = tmp_path / "store" / "pub-a" / "AA"
    store.mkdir(parents=True)
    (store / "AApano1.jpg").write_bytes(b"x")
    present = cc.present_in_store(files, str(tmp_path / "store"))
    assert present == {"pub-a/AA/AApano1.jpg"}


# ---------------------------------------------------------------------------------------------------------------------
# recrop
# ---------------------------------------------------------------------------------------------------------------------

def test_recrop_cuts_three_crops_from_one_render_sized_from_the_recorded_depth(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    render, reader = run_recrop(root, frame, work, max_labels=6)
    rec = cc.load_recrop(work)
    assert rec.label_uid.is_unique and set(rec.label_uid) == set(frame.label_uid)
    ok = rec[rec.status == "ok"].merge(frame[["label_uid", "depth_m", "crop_half_px", "crop_path"]],
                                       on="label_uid", validate="one_to_one")
    assert len(render.calls) == len(ok)                             # one render per label, three crops from it
    read_panos = [(c, p) for _, c, p in reader.calls]
    assert len(read_panos) == len(set(read_panos))                  # one read per panorama
    assert (ok.half_4000 == (4000 / ok.depth_m).astype(int)).all()
    assert (ok.half_8000 == (8000 / ok.depth_m).astype(int)).all()
    assert (ok.half_6100 == ok.crop_half_px).all()                  # the recorded size, not a recomputed one
    first = ok.sort_values("label_uid").iloc[0]
    assert (ok.half_6100_recut == (6100 / ok.depth_m).astype(int)).all()
    assert (ok.half_6100 != ok.half_6100_recut).sum() == 1          # the planted +1 row
    for c in cc.ARMS:
        for r in ok.itertuples():
            assert os.path.exists(os.path.join(cc.arm_dir(work, c), r.label_type, r.crop_path))
    assert (ok.parity_mad < 0.5).all()                              # same view, same encode: near-identical
    assert first.renderer == "numpy"


def test_recrop_resumes_without_rendering_again(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    before = cc.load_recrop(work)
    render, _ = run_recrop(root, frame, work)
    assert render.calls == []
    pd.testing.assert_frame_equal(cc.load_recrop(work), before)


def test_recrop_retries_errors_and_drops_a_cut_off_line(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    boom = FakeRender()

    def flaky(*a, **k):
        if len(boom.calls) == 0:
            boom.calls.append("boom")
            raise RuntimeError("transient")
        return FakeRender()(*a, **k)
    run_recrop(root, frame, work, render=flaky)
    rec = cc.load_recrop(work)
    assert rec.status.str.startswith("error:").sum() == 1
    path = os.path.join(work, "recrop", "chunk_0000.csv")
    with open(path, "a") as f:
        f.write("pub-a:9999,CurbRamp,o")                            # a write cut off by a preemption
    run_recrop(root, frame, work)
    rec = cc.load_recrop(work)
    assert not rec.status.str.startswith("error:").any() and "pub-a:9999" not in set(rec.label_uid)


def test_recrop_uses_crop_pose_and_centre_crop_not_an_inline_formula(r3, tmp_path, monkeypatch):
    root, frame = r3
    calls = {"rig": 0, "centre": 0, "pose": 0}
    real_rig, real_centre, real_pose = crop_pose.rig_y_norm, dg.centre_crop, crop_pose.pose_for_row

    def rig(y, pose):
        calls["rig"] += 1
        return real_rig(y, pose)

    def centre(view, half):
        calls["centre"] += 1
        return real_centre(view, half)

    def pose(row, beta, rot=None):
        calls["pose"] += 1
        assert beta == 0.95
        return real_pose(row, beta, rot)
    monkeypatch.setattr(crop_pose, "rig_y_norm", rig)
    monkeypatch.setattr(dg, "centre_crop", centre)
    monkeypatch.setattr(crop_pose, "pose_for_row", pose)
    run_recrop(root, frame, str(tmp_path / "w"))
    n_ok = int((cc.load_recrop(str(tmp_path / "w")).status == "ok").sum())
    assert calls["rig"] >= n_ok and calls["pose"] >= n_ok
    assert calls["centre"] == 3 * n_ok                              # 6100 directly, 4000 and 8000 via depth_sized_crop


def test_recrop_passes_the_pose_to_the_renderer_as_data_generation_does(r3, tmp_path):
    root, frame = r3
    render, _ = run_recrop(root, frame, str(tmp_path / "w"))
    x, y, pitch, roll, tilt, beta, device = render.calls[0]
    assert pitch == 1.25 and roll is None and tilt is True and beta == 0.95 and device is None


def test_pano_store_override_and_recorded_store_root(r3, tmp_path):
    root, frame = r3
    _, reader = run_recrop(root, frame, str(tmp_path / "a"))
    assert {s for s, _, _ in reader.calls} == {"/stores/r3/CurbRamp", "/stores/r3/SurfaceProblem"}
    _, reader = run_recrop(root, frame, str(tmp_path / "b"), pano_store="/scrubbed/store")
    assert {s for s, _, _ in reader.calls} == {"/scrubbed/store"}
    rec = cc.load_recrop(str(tmp_path / "b"))
    assert set(rec.store_root) == {"/stores/r3/CurbRamp", "/stores/r3/SurfaceProblem"}   # parity reported per store


def test_a_native_size_mismatch_and_a_missing_panorama_are_lost(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    rec = cc.load_recrop(work).set_index("label_uid")
    small = frame[frame.pano_id.str.startswith("SMALL")].label_uid
    miss = frame[frame.pano_id.str.startswith("MISS")].label_uid
    assert (rec.loc[small, "status"] == "native_mismatch").all() and not rec.loc[small, "native_match"].any()
    assert (rec.loc[miss, "status"] == "no_pano").all()
    for c in cc.ARMS:                                                 # no crop for a lost row in any arm
        for r in frame[frame.label_uid.isin(set(small) | set(miss))].itertuples():
            assert not os.path.exists(os.path.join(cc.arm_dir(work, c), r.label_type, r.crop_path))


def test_a_missing_stored_crop_fails_parity_rather_than_passing_it(r3, tmp_path):
    root, frame = r3
    victim = frame[frame.status == "ok"].iloc[0]
    os.remove(os.path.join(root, victim.label_type, victim.crop_path))
    run_recrop(root, frame, str(tmp_path / "w"))
    rec = cc.load_recrop(str(tmp_path / "w")).set_index("label_uid")
    assert np.isinf(rec.loc[victim.label_uid, "parity_mad"])


# ---------------------------------------------------------------------------------------------------------------------
# Gates and the arm directories
# ---------------------------------------------------------------------------------------------------------------------

def _rec(n, lost=(), split=None, types=("CurbRamp",)):
    rows = []
    for t in types:
        for i in range(n):
            s = (split or (lambda i: "train"))(i)
            rows.append({"label_uid": f"c:{t}{i}", "label_type": t, "split": s,
                         "status": "no_pano" if i in lost else "ok"})
    return pd.DataFrame(rows)


def test_g2_row_loss_boundaries():
    def split(i):
        return "test" if i < 100 else "train"
    rec = _rec(1000, lost={500, 501, 502, 503, 504}, split=split)      # 0.5 % of the type: passes
    frame = rec[["label_uid", "label_type", "split"]]
    assert cc.gate_g2(frame, rec)["passed"]
    rec = _rec(1000, lost={500, 501, 502, 503, 504, 505}, split=split)  # 0.6 %
    assert not cc.gate_g2(frame, rec)["passed"]
    rec = _rec(1000, lost={0}, split=split)                             # 1 of 100 test rows: 1 %, passes
    assert cc.gate_g2(frame, rec)["passed"]
    rec = _rec(1000, lost={0, 1}, split=split)                          # 2 % of test
    g = cc.gate_g2(frame, rec)
    assert not g["passed"] and not g["by_type"]["CurbRamp"]["test"]["passed"]


def test_g2_counts_held_out_rows_and_refuses_an_incomplete_recrop():
    def split(i):
        return "test_heldout" if i < 50 else "train"
    rec = _rec(1000, lost={0}, split=split)                              # 1 of 50 held-out rows: 2 %
    frame = rec[["label_uid", "label_type", "split"]]
    assert not cc.gate_g2(frame, rec)["passed"]
    g = cc.gate_g2(frame, rec.iloc[1:])                                  # a row never attempted
    assert not g["passed"] and g["not_attempted"] == 1


def test_g2_refuses_a_row_never_attempted_even_within_the_allowance():
    rec = _rec(1000)
    frame = rec[["label_uid", "label_type", "split"]]
    g = cc.gate_g2(frame, rec.iloc[1:])                                  # 0.1 % of the type, all of it unattempted
    assert g["by_type"]["CurbRamp"]["passed"] and g["not_attempted"] == 1 and not g["passed"]


def test_load_recrop_prefers_an_ok_row_over_a_stale_error_in_a_later_file(tmp_path):
    work = str(tmp_path)
    os.makedirs(os.path.join(work, "recrop"))
    base = {f: "" for f in cc.RECROP_FIELDS}
    rows = {0: [{**base, "label_uid": "c:1", "status": "ok"}, {**base, "label_uid": "c:2", "status": "error:x"}],
            1: [{**base, "label_uid": "c:1", "status": "error:old"}, {**base, "label_uid": "c:2", "status": "ok"},
                {**base, "label_uid": "c:3", "status": "error:a"}, {**base, "label_uid": "c:3", "status": "no_pano"}]}
    for k, rr in rows.items():
        pd.DataFrame(rr, columns=cc.RECROP_FIELDS).to_csv(cc.chunk_path(work, k), index=False)
    rec = cc.load_recrop(work).set_index("label_uid").status
    assert rec.to_dict() == {"c:1": "ok", "c:2": "ok", "c:3": "no_pano"}


# ---------------------------------------------------------------------------------------------------------------------
# preflight (G0) and the gates command
# ---------------------------------------------------------------------------------------------------------------------

def _store(tmp_path, panos, missing=()):
    store = str(tmp_path / "store")
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, dg.PANO_STORE_MARKER), "w") as f:
        f.write("test store\n")
    for c, p in panos:
        if p not in missing:
            path = dg.pano_store_path(store, c, p)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
    return store


def _preflight(root, store, **kw):
    pins, _ = cc.da.input_pins(root, os.path.join(root, "rawlabels"), ("CurbRamp",))
    args = {"package": lambda: "pkg@abc", "versions": {k: "1" for k in cc.REQUIRED_VERSIONS}, "expected_pins": pins}
    args.update(kw)
    return cc.preflight(root, os.path.join(root, "rawlabels"), ("CurbRamp",), store, **args)


@pytest.fixture
def r3_400(tmp_path):
    root = str(tmp_path / "r3")
    panos = {"CurbRamp": [("pub-a", f"AA{i:04d}") for i in range(400)]}
    make_round3(root, panos=panos, labels_per_pano=1, write_stored=False)
    return root, panos["CurbRamp"]


def test_preflight_passes_a_matching_round3_and_store(r3_400, tmp_path):
    root, panos = r3_400
    rec = _preflight(root, _store(tmp_path, panos))
    assert rec["g0"]["passed"] and rec["store"]["passed"] and rec["store"]["missing_panoramas"] == 0
    assert rec["package_source"] == {"now": "pkg@abc", "round3": {"CurbRamp": "pkg@abc"}}


def test_preflight_refuses_a_pin_mismatch(r3_400, tmp_path):
    root, panos = r3_400
    pins, _ = cc.da.input_pins(root, os.path.join(root, "rawlabels"), ("CurbRamp",))
    bad = {k: ("0" * 64 if i == 0 else v) for i, (k, v) in enumerate(pins.items())}
    rec = _preflight(root, _store(tmp_path, panos), expected_pins=bad)
    assert not rec["g0"]["passed"] and "pins" in rec["g0"]["failures"][0]


def test_preflight_refuses_another_panorama_tools_commit(r3_400, tmp_path):
    root, panos = r3_400
    rec = _preflight(root, _store(tmp_path, panos), package=lambda: "pkg@def")
    assert not rec["g0"]["passed"] and "sidewalk-panorama-tools" in rec["g0"]["failures"][0]


def test_preflight_refuses_unrecorded_versions(r3_400, tmp_path):
    root, panos = r3_400
    rec = _preflight(root, _store(tmp_path, panos), versions={"torch": "2.5.1"})
    assert not rec["g0"]["passed"] and "versions not recorded" in rec["g0"]["failures"][0]


def test_preflight_tolerates_missing_panoramas_only_within_g2(r3_400, tmp_path):
    root, panos = r3_400
    one = _preflight(root, _store(tmp_path / "a", panos, missing={panos[0][1]}))   # 1 of 400 rows: 0.25 %
    assert one["store"]["passed"] and one["store"]["missing_panoramas"] == 1
    assert one["store"]["missing_rows"]["CurbRamp"]["lost"] == 1
    three = _preflight(root, _store(tmp_path / "b", panos, missing={p for _, p in panos[:3]}))   # 0.75 %
    assert not three["store"]["passed"] and "break G2" in three["store"]["failures"][0]
    assert three["g0"]["passed"]                                         # the store fails, not G0


def test_preflight_refuses_a_store_without_its_marker(r3_400, tmp_path):
    root, panos = r3_400
    store = _store(tmp_path, panos)
    os.remove(os.path.join(store, dg.PANO_STORE_MARKER))
    assert not _preflight(root, store)["store"]["passed"]


def test_the_preflight_command_exits_non_zero_on_a_failure(r3_400, tmp_path, monkeypatch):
    root, panos = r3_400
    store = _store(tmp_path, panos, missing={p for _, p in panos[:3]})
    rec = _preflight(root, store)                       # the real check, with this round 3's pins and package
    monkeypatch.setattr(cc, "preflight", lambda *a, **k: rec)
    out = str(tmp_path / "preflight.json")
    with pytest.raises(SystemExit) as e:
        cc.main(["preflight", "--root", root, "--rawlabels-dir", os.path.join(root, "rawlabels"), "--types",
                 "CurbRamp", "--pano-store", store, "--out", out])
    assert e.value.code and "preflight failed" in str(e.value.code)
    with open(out) as f:
        assert not json.load(f)["store"]["passed"]


def _gates(root, work, g0=True):
    os.makedirs(work, exist_ok=True)
    cc.write_json(os.path.join(work, "preflight.json"), {"g0": {"passed": g0, "failures": []},
                                                        "store": {"pano_store": "/s"}})
    return ["gates", "--root", root, "--rawlabels-dir", os.path.join(root, "rawlabels"), "--types",
            ",".join(TYPES), "--work", work]


def test_a_failed_gate_writes_no_arm_directory_and_exits_non_zero(r3, tmp_path):
    root, frame = r3                                     # MISS and SMALL panoramas: far above G2's allowance
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    with pytest.raises(SystemExit) as e:
        cc.main(_gates(root, work))
    assert "no arm directory" in str(e.value.code)
    assert not any(os.path.exists(os.path.join(cc.arm_dir(work, c), t, f)) for c in cc.ARMS for t in TYPES
                   for f in ("split.csv", "manifest.csv", "generation_run.json"))   # the crops alone exist
    with open(os.path.join(work, "gates.json")) as f:
        g = json.load(f)
    assert not g["passed"] and not g["G2"]["passed"] and "arms" not in g


def test_the_gates_command_writes_the_arms_only_when_every_gate_passes(tmp_path):
    root = str(tmp_path / "r3")
    panos = {t: [("pub-a", f"{t[:2].upper()}{i:03d}") for i in range(6)] for t in TYPES}
    frame = make_round3(root, panos=panos)
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    with pytest.raises(SystemExit):                      # G0 failed in preflight: still nothing written
        cc.main(_gates(root, work, g0=False))
    assert not any(os.path.exists(os.path.join(cc.arm_dir(work, c), t, "split.csv")) for c in cc.ARMS for t in TYPES)
    cc.main(_gates(root, work))
    with open(os.path.join(work, "gates.json")) as f:
        g = json.load(f)
    assert g["passed"] and g["G2"]["passed"] and g["G3"]["passed"]
    for c in cc.ARMS:
        for t in TYPES:
            assert os.path.exists(os.path.join(cc.arm_dir(work, c), t, "split.csv"))


def _parity(mads, store="/s/a"):
    return pd.DataFrame({"label_uid": [f"c:{i}" for i in range(len(mads))], "label_type": "CurbRamp",
                         "status": "ok", "parity_mad": mads, "store_root": store})


def test_g3_parity_boundaries():
    assert cc.gate_g3(_parity([3.0] * 1000))["passed"]                  # exactly 3.0 is within
    assert cc.gate_g3(_parity([1.0] * 995 + [12.0] * 5))["passed"]      # 99.5 % within, none above 12
    assert not cc.gate_g3(_parity([1.0] * 994 + [12.0] * 6))["passed"]
    assert not cc.gate_g3(_parity([1.0] * 999 + [12.01]))["passed"]     # one row above 12
    assert not cc.gate_g3(_parity([1.0] * 999 + [np.inf]))["passed"]    # no stored crop: unjudged is a failure
    assert not cc.gate_g3(_parity([1.0] * 999 + [np.nan]))["passed"]
    two = pd.concat([_parity([1.0] * 10, "/s/a"), _parity([20.0], "/s/b")], ignore_index=True)
    g = cc.gate_g3(two)
    assert not g["passed"] and set(g["by_store"]) == {"/s/a", "/s/b"} and g["by_store"]["/s/b"]["above_hard"] == 1


def test_arms_keep_round3_rows_order_and_split_and_drop_a_lost_row_everywhere(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    rec = cc.load_recrop(work)
    lost = set(rec.label_uid[rec.status != "ok"])
    assert lost
    cc.write_arms(work, root, frame, rec, TYPES, {"tool": "test"})
    for t in TYPES:
        src = pd.read_csv(os.path.join(root, t, "split.csv"), dtype=str, keep_default_na=False)
        kept = src[~src.label_uid.isin(lost)].reset_index(drop=True)
        for c in cc.ARMS:
            arm = pd.read_csv(os.path.join(cc.arm_dir(work, c), t, "split.csv"), dtype=str, keep_default_na=False)
            assert arm.label_uid.tolist() == kept.label_uid.tolist()          # same rows, same order
            m = arm.merge(kept, on="label_uid", suffixes=("", "_r3"), validate="one_to_one")
            assert (m.split == m.split_r3).all() and (m.depth_m == m.depth_m_r3).all()
            for col in set(SPLIT_COLS) - {"label_uid", "crop_half_px", "clipped"}:
                assert (m[col] == m[col + "_r3"]).all(), col                  # only the size columns change
            expect = {6100: m.crop_half_px_r3.astype(int),
                      4000: (4000 / m.depth_m.astype(float)).astype(int),
                      8000: (8000 / m.depth_m.astype(float)).astype(int)}[c]
            assert (m.crop_half_px.astype(int) == expect).all()
            manifest = pd.read_csv(os.path.join(cc.arm_dir(work, c), t, "manifest.csv"), dtype=str)
            assert manifest.label_uid.tolist() == arm.label_uid.tolist() and (manifest.status == "ok").all()
            assert "split" not in manifest.columns
        assert os.path.exists(os.path.join(cc.arm_dir(work, 4000), "deployment_priors.json"))


def test_the_arm_record_keeps_round3s_and_adds_the_constant_and_source(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    cc.write_arms(work, root, frame, cc.load_recrop(work), TYPES, {"tool": "test"})
    src_sha = cc.sha256_file(os.path.join(root, "CurbRamp", "split.csv"))
    for c in cc.ARMS:
        with open(os.path.join(cc.arm_dir(work, c), "CurbRamp", "generation_run.json")) as f:
            run = json.load(f)
        assert run["tilt_beta"] == 0.95 and run["depth_encoder"] == "vitb"
        assert run["settings"]["depth_encoder"] == "vitb"
        assert run["crop_depth_constant"] == c and run["settings"]["crop_depth_constant"] == c
        assert run["derived_from"]["split_sha256"] == src_sha and run["recrop"] == {"tool": "test"}
    import fine_tune_classifier as ft
    s = ft.generation_settings("CurbRamp", cc.arm_dir(work, 4000))
    assert s == {"tilt_beta": 0.95, "depth_encoder": "vitb", "crop_depth_constant": 4000}


def test_the_6100_arm_passes_the_publish_guard_and_the_others_do_not(r3, tmp_path):
    import publish_to_hf as pub
    root, frame = r3
    work = str(tmp_path / "w")
    run_recrop(root, frame, work)
    cc.write_arms(work, root, frame, cc.load_recrop(work), TYPES, {"tool": "test"})
    for c in (4000, 8000):
        with open(os.path.join(cc.arm_dir(work, c), "CurbRamp", "generation_run.json")) as f:
            with pytest.raises(pub.PublishError, match=f"records crop_depth_constant {c}"):
                pub.check_crop_constant(json.load(f), "g.json")
        arm = pd.read_csv(os.path.join(cc.arm_dir(work, c), "CurbRamp", "split.csv"))
        with pytest.raises(pub.PublishError, match="rows were not cut"):
            pub.check_depth_rows(arm, "split.csv")
    arm = pd.read_csv(os.path.join(cc.arm_dir(work, 6100), "CurbRamp", "split.csv"))
    pub.check_depth_rows(arm, "split.csv")


# ---------------------------------------------------------------------------------------------------------------------
# Statistics: the paired, seed-paired bootstrap
# ---------------------------------------------------------------------------------------------------------------------

def _scores(n_pano=150, per=2, k=3, seed=0, gain=0.0, gain_by_seed=None):
    rng = np.random.default_rng(seed)
    n = n_pano * per
    y = np.zeros(n, bool)
    y[rng.permutation(n)[: n // 2]] = True
    clusters = np.repeat([f"c:p{i}" for i in range(n_pano)], per)
    base = np.stack([y + rng.normal(0, 1.0, n) for _ in range(k)])
    g = gain_by_seed if gain_by_seed is not None else [gain] * k
    arm = np.stack([y * (1 + g[s]) + rng.normal(0, 1.0, n) for s in range(k)])
    return y, arm, base, clusters


def test_weighted_auc_matches_sklearn_and_repeated_rows():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.random(200) < 0.4
    s = np.round(rng.normal(size=200), 1)                             # ties included
    r = cc.RankedScores(s)
    assert abs(r.auc(y, np.ones(200)) - roc_auc_score(y, s)) < 1e-12
    w = rng.integers(0, 3, 200)
    idx = np.repeat(np.arange(200), w)
    assert abs(r.auc(y, w.astype(float)) - roc_auc_score(y[idx], s[idx])) < 1e-12


def test_identical_predictions_give_exactly_zero():
    y, _, base, clusters = _scores(k=3)
    st = cc.paired_stats(y, base.copy(), base, clusters, k=3, n_boot=200, seed=1)
    assert st["point"] == 0 and st["reg_lo"] == 0 and st["reg_hi"] == 0
    assert st["hier_lo"] == 0 and st["hier_hi"] == 0 and st["seed_deltas"] == [0, 0, 0]
    assert cc.arm_passes({**st, "unestimable": None, "heldout_within": 0.0})[0] is False   # a bound of 0 is not > 0


def test_arm_and_baseline_share_each_panorama_draw():
    """D[b, s] must come from one panorama draw for both models: with arm == baseline every replicate is exactly 0."""
    y, _, base, clusters = _scores(k=3)
    D, draws = cc.replicates(y, base.copy(), base, clusters, n_boot=100, seed=2)
    assert D.shape == (100, 3) and (D == 0).all()


def test_all_drawn_seeds_share_one_panorama_draw():
    """Swap the two models between seeds: with one panorama draw per replicate, seed 1's delta is exactly minus seed
    0's in every replicate. Independent draws per seed would break the identity."""
    y, arm, base, clusters = _scores(k=2, gain=0.3)
    arm2 = np.stack([arm[0], base[0]])
    base2 = np.stack([base[0], arm[0]])
    D, draws = cc.replicates(y, arm2, base2, clusters, n_boot=200, seed=3)
    assert np.allclose(D[:, 0], -D[:, 1], atol=1e-15) and np.abs(D[:, 0]).max() > 0
    st = cc.paired_stats(y, arm2, base2, clusters, k=2, n_boot=200, seed=3)
    assert abs(st["reg_lo"]) < 1e-15 and abs(st["reg_hi"]) < 1e-15   # registered: the mean over all seeds is 0
    hier = np.array([D[b, draws[b]].mean() for b in range(len(D))])    # hierarchical: the drawn seeds, same draw
    lo, hi = np.percentile(hier, [100 * cc.ALPHA / cc.M_ARMS / 2, 100 - 100 * cc.ALPHA / cc.M_ARMS / 2])
    assert (st["hier_lo"], st["hier_hi"]) == (lo, hi)


def test_a_planted_gain_passes():
    y, arm, base, clusters = _scores(k=3, gain=1.0, n_pano=300)
    st = cc.paired_stats(y, arm, base, clusters, k=3, n_boot=300, seed=4)
    assert st["point"] > 0.05 and st["reg_lo"] > 0 and st["hier_lo"] > 0 and all(d > 0 for d in st["seed_deltas"])
    assert cc.arm_passes({**st, "unestimable": None, "heldout_within": 0.01})[0]
    assert st["level"] == 1 - 0.05 / 2


def test_a_gain_carried_by_one_seed_fails():
    y, arm, base, clusters = _scores(k=3, gain_by_seed=[2.0, -0.05, -0.05], n_pano=300)
    st = cc.paired_stats(y, arm, base, clusters, k=3, n_boot=300, seed=5)
    assert st["point"] > 0
    ok, why = cc.arm_passes({**st, "unestimable": None, "heldout_within": 0.01})
    assert not ok and not cc.seed_guard(st["seed_deltas"], 3)       # the guard fails too, whatever the intervals say
    assert not cc.arm_passes({**st, "reg_lo": 0.01, "hier_lo": 0.01, "unestimable": None,
                              "heldout_within": 0.01})[0]


def test_mdd_is_the_delta_both_lower_bounds_needed():
    y, arm, base, clusters = _scores(k=3, gain=0.05)
    st = cc.paired_stats(y, arm, base, clusters, k=3, n_boot=200, seed=6)
    assert st["mdd"] == pytest.approx(st["point"] - min(st["reg_lo"], st["hier_lo"]))


@pytest.mark.parametrize("deltas, k, ok", [
    ([0.01, 0.02, 0.03], 3, True), ([0.01, 0.02, -0.01], 3, False), ([0.01, 0.02, 0.0], 3, False),
    ([0.01, 0.02, 0.03, 0.04, -0.01], 5, True), ([0.01, 0.02, 0.03, -0.04, -0.01], 5, False),
    ([0.01] * 5, 5, True), ([0.01, 0.02, 0.03, 0.01], 4, True), ([0.01, 0.02, 0.03, -0.01], 4, False)])
def test_seed_guard_by_registered_seed_count(deltas, k, ok):
    assert cc.seed_guard(deltas, k) is ok


def test_seed_guard_refuses_a_list_of_another_length():
    assert not cc.seed_guard([0.01, 0.02, 0.03, 0.04], 5) and not cc.seed_guard([0.01] * 4, 3)


# ---------------------------------------------------------------------------------------------------------------------
# The read-out (R3 with the three amendments)
# ---------------------------------------------------------------------------------------------------------------------

def cell(point=0.01, reg_lo=0.002, hier_lo=0.001, seeds=(0.01, 0.01, 0.01), within=0.0, unestimable=None):
    return {"point": point, "reg_lo": reg_lo, "reg_hi": point + 0.01, "hier_lo": hier_lo, "hier_hi": point + 0.01,
            "seed_deltas": list(seeds), "k": len(seeds), "heldout_within": within, "unestimable": unestimable,
            "mdd": point - min(reg_lo, hier_lo)}


def fail():
    return cell(point=0.0, reg_lo=-0.01, hier_lo=-0.01, seeds=(0.0, 0.01, -0.01))


def test_a_lower_bound_of_exactly_zero_does_not_pass():
    assert not cc.arm_passes(cell(reg_lo=0.0))[0]
    assert not cc.arm_passes(cell(hier_lo=0.0))[0]
    assert cc.arm_passes(cell())[0]


def test_one_interval_above_zero_and_the_other_not_does_not_pass():
    assert not cc.arm_passes(cell(reg_lo=0.002, hier_lo=-0.001))[0]
    assert not cc.arm_passes(cell(reg_lo=-0.001, hier_lo=0.002))[0]


def test_the_held_out_within_city_clause():
    assert cc.arm_passes(cell(within=0.0))[0]                            # at least 0, as registered
    assert not cc.arm_passes(cell(within=-1e-9))[0]
    ok, why = cc.arm_passes(cell(within=None))
    assert not ok and "held-out" in why


def test_an_unestimable_cell_never_reads_as_pass_in_its_verdict():
    assert cc.verdict(cell(unestimable="seed 43 has no prediction for 1 of 120 rows")) == "unestimable"
    assert cc.verdict(cell(point=-0.03, reg_lo=-0.05, hier_lo=-0.05, unestimable="x")) == "unestimable"


def test_unestimable_is_never_a_pass():
    ok, why = cc.arm_passes(cell(unestimable="fewer than the 5 registered seeds"))
    assert not ok and "unestimable" in why


def test_the_conjunction_over_types_and_stage_one_outcomes():
    r = cc.rule_readout({"CurbRamp": {4000: cell(), 8000: fail()}, "SurfaceProblem": {4000: cell(), 8000: fail()}})
    assert r["outcome"] == "candidate" and r["passing_arms"] == [4000] and r["stage2_triggered"]
    r = cc.rule_readout({"CurbRamp": {4000: cell(), 8000: fail()}, "SurfaceProblem": {4000: fail(), 8000: fail()}})
    assert r["outcome"] == "no_change" and r["passing_arms"] == [] and not r["stage2_triggered"]
    r = cc.rule_readout({"CurbRamp": {4000: cell()}})                    # CurbRamp only: a candidate at most
    assert r["outcome"] == "candidate" and not r["stage2_triggered"]


def test_both_arms_passing_changes_nothing():
    r = cc.rule_readout({"CurbRamp": {4000: cell(), 8000: cell()}, "SurfaceProblem": {4000: cell(), 8000: cell()}})
    assert r["outcome"] == "both_pass" and not r["stage2_triggered"] and r["passing_arms"] == [4000, 8000]
    assert "not read as a constant effect" in r["reading"]


def test_stage_two_reads_one_arm_with_m_two_and_can_change_the_default():
    five = {t: {8000: cell()} for t in cc.STAGE2_TYPES}
    five.update({t: {4000: fail(), 8000: cell()} for t in cc.STAGE1_TYPES})
    r = cc.rule_readout(five)
    assert r["outcome"] == "change" and r["passing_arms"] == [8000] and r["m"] == 2
    five["Obstacle"][8000] = fail()
    assert cc.rule_readout(five)["outcome"] == "no_change"


def test_stage_two_statistics_use_m_two_even_with_one_arm_trained():
    y, arm, base, clusters = _scores(k=3, gain=0.2)
    st = cc.paired_stats(y, arm, base, clusters, k=3, n_boot=200, seed=7)
    assert st["level"] == 0.975 and st["m"] == 2


def test_an_unestimable_cell_blocks_a_reading():
    r = cc.rule_readout({"CurbRamp": {4000: cell(unestimable="family differs: git_commit"), 8000: fail()},
                         "SurfaceProblem": {4000: cell(), 8000: fail()}})
    assert r["outcome"] == "unestimable" and r["passing_arms"] == []
    assert any("git_commit" in u for u in r["unestimable"])


def test_a_significantly_worse_arm_is_reported_as_worse():
    worse = cell(point=-0.03, reg_lo=-0.05, hier_lo=-0.06, seeds=(-0.03, -0.02, -0.04), within=-0.02)
    worse.update(reg_hi=-0.01, hier_hi=-0.005)
    r = cc.rule_readout({"CurbRamp": {4000: worse, 8000: fail()}, "SurfaceProblem": {4000: fail(), 8000: fail()}})
    assert r["verdicts"]["CurbRamp"][4000] == "worse"


def test_the_few_seeds_caveat_is_printed():
    r = cc.rule_readout({"CurbRamp": {4000: cell(), 8000: fail()}})
    assert "3.7 %" in r["caveat"]
    five = cell(seeds=(0.01,) * 5)
    assert "0.032 %" in cc.rule_readout({"CurbRamp": {4000: five, 8000: {**five, "reg_lo": -0.1}}})["caveat"]


# ---------------------------------------------------------------------------------------------------------------------
# compare: the family, the registered seeds, the classes
# ---------------------------------------------------------------------------------------------------------------------

def make_family(work, t="CurbRamp", arms=cc.ARMS, seeds=(42, 43, 44, 45, 46), gain=0.0, n_pano=120, seed=0,
                versions=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pano):
        city = ["pub-a", "pub-b", "secret-town"][i % 3]
        split = "test" if i < 60 else ("test_heldout" if i < 96 else "train")
        hcity = ["held-x", "held-y", "secret-held"][i % 3] if split == "test_heldout" else city
        for j in range(2):
            n = len(rows)
            rows.append({"label_uid": f"{hcity}:{n}", "city": hcity, "label_id": str(n), "pano_id": f"p{i}",
                         "label_class": "incorrect" if (i + j) % 2 else "correct", "split": split,
                         "severity": 1 + n % 3, "tags": '["narrow"]' if n % 4 else "[]",
                         "image_capture_date": "2019-05", "crop_half_px": 200 + 7 * (n % 50), "depth_m": 10.0,
                         "crop_path": f"x/{n}.webp", "status": "ok"})
    split = pd.DataFrame(rows)
    y = (split.label_class == "incorrect").to_numpy()
    for c in arms:
        d = os.path.join(cc.arm_dir(work, c), t)
        os.makedirs(d, exist_ok=True)
        split.to_csv(os.path.join(d, "split.csv"), index=False)
        for s in seeds:
            g = gain if c != cc.BASELINE else 0.0
            score = 1 / (1 + np.exp(-(y * (1 + g) + rng.normal(0, 1, len(y)))))
            pd.DataFrame({"label_uid": split.label_uid, "split": split.split, "true": split.label_class,
                          "p_correct": 1 - score, "p_incorrect": score}).to_csv(
                os.path.join(d, f"predictions_s{s}.csv"), index=False)
            rd = os.path.join(work, "runs", cc.arm_name(c), t, f"s{s}")
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, "training_run.json"), "w") as f:
                json.dump({"label_type": t, "seed": s, "git_commit": "a" * 40, "git_dirty": False,
                           "crop_depth_constant": c, "versions": versions or {"torch": "2.5.1", "numpy": "2.1"}}, f)
    return split


def _edit_run(work, c, s, **kv):
    p = os.path.join(work, "runs", cc.arm_name(c), "CurbRamp", f"s{s}", "training_run.json")
    with open(p) as f:
        run = json.load(f)
    run.update(kv)
    with open(p, "w") as f:
        json.dump(run, f)


def test_compare_reads_a_complete_family(tmp_path):
    work = str(tmp_path)
    make_family(work, gain=0.8)
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=100)
    for c in (4000, 8000):
        assert res["cells"][c]["unestimable"] is None and res["cells"][c]["k"] == 5
        assert res["cells"][c]["point"] > 0
    assert res["family_problems"] == []
    assert len(res["inputs"]) == 3 * (1 + 2 * 5)                       # every split, prediction and run record
    assert res["inputs"]["c4000/CurbRamp/predictions_s42.csv"] == cc.sha256_file(
        os.path.join(cc.arm_dir(work, 4000), "CurbRamp", "predictions_s42.csv"))


def test_four_of_five_registered_seeds_is_unestimable(tmp_path):
    work = str(tmp_path)
    make_family(work)
    os.remove(os.path.join(cc.arm_dir(work, 4000), "CurbRamp", "predictions_s45.csv"))
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=50)
    assert "registered" in res["cells"][4000]["unestimable"] and res["cells"][8000]["unestimable"] is None
    shutil.rmtree(os.path.join(work, "runs", "c6100", "CurbRamp", "s46"))   # a baseline seed gone: both arms
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=50)
    assert res["cells"][4000]["unestimable"] and "registered" in res["cells"][8000]["unestimable"]


@pytest.mark.parametrize("edit, word", [({"git_commit": "b" * 40}, "git_commit"), ({"git_dirty": True}, "git_dirty"),
                                        ({"versions": {"torch": "2.6.0", "numpy": "2.1"}}, "versions"),
                                        ({"git_state_at_finish": {"git_commit": "c" * 40, "git_dirty": False}},
                                         "git_state_at_finish")])
def test_a_family_that_differs_in_code_or_packages_is_unestimable(tmp_path, edit, word):
    work = str(tmp_path)
    make_family(work, gain=0.8)
    _edit_run(work, 6100, 44, **edit)
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=50)
    assert any(word in p for p in res["family_problems"])
    for c in (4000, 8000):
        assert word in res["cells"][c]["unestimable"]


def test_a_run_trained_on_another_arm_is_caught(tmp_path):
    work = str(tmp_path)
    make_family(work)
    _edit_run(work, 8000, 42, crop_depth_constant=6100)
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=50)
    assert "crop_depth_constant" in res["cells"][8000]["unestimable"]


@pytest.mark.parametrize("value", [None, "8000x", "absent"])
def test_a_run_recording_no_usable_crop_constant_is_unestimable(tmp_path, value):
    work = str(tmp_path)
    make_family(work, gain=0.8)
    for s in (42, 43, 44, 45, 46):
        p = os.path.join(work, "runs", "c8000", "CurbRamp", f"s{s}", "training_run.json")
        with open(p) as f:
            run = json.load(f)
        if value == "absent":
            run.pop("crop_depth_constant")
        else:
            run["crop_depth_constant"] = value
        with open(p, "w") as f:
            json.dump(run, f)
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=30)
    assert "crop_depth_constant" in res["cells"][8000]["unestimable"] and res["cells"][4000]["unestimable"] is None


@pytest.mark.parametrize("seeds", [(42, 43, 44), (42, 43, 45), (42, 43, 44, 45)])
def test_a_reading_on_a_subset_of_the_trained_seeds_is_unestimable(tmp_path, seeds):
    work = str(tmp_path)
    make_family(work, gain=0.8)                                        # seeds 42 to 46 trained and on disk
    res = cc.compare_type(work, "CurbRamp", seeds=seeds, n_boot=30)
    for c in (4000, 8000):
        assert "outside the registered" in res["cells"][c]["unestimable"]
    shutil.rmtree(os.path.join(work, "runs", "c8000", "CurbRamp", "s46"))   # a run record alone still counts
    os.remove(os.path.join(cc.arm_dir(work, 8000), "CurbRamp", "predictions_s46.csv"))
    os.makedirs(os.path.join(work, "runs", "c8000", "CurbRamp", "s47"))
    cc.write_json(os.path.join(work, "runs", "c8000", "CurbRamp", "s47", "training_run.json"), {"seed": 47})
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45), n_boot=30)
    why = res["cells"][8000]["unestimable"]
    assert "c8000 holds predictions or runs for seeds outside the registered ones (s47)" in why
    assert "baseline: c6100 holds" in why and "(s46)" in why             # c6100 still holds s46


def test_an_unregistered_seed_is_found_from_a_predictions_file_alone(tmp_path):
    work = str(tmp_path)
    make_family(work, gain=0.8)
    shutil.copy(os.path.join(cc.arm_dir(work, 4000), "CurbRamp", "predictions_s46.csv"),
                os.path.join(cc.arm_dir(work, 4000), "CurbRamp", "predictions_s47.csv"))   # no run directory
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=30)
    assert "outside the registered ones (s47)" in res["cells"][4000]["unestimable"]
    assert res["cells"][8000]["unestimable"] is None


def test_an_error_that_is_not_unestimable_stops_compare(tmp_path, monkeypatch):
    work = str(tmp_path)
    make_family(work, seeds=(42, 43, 44), gain=0.8)

    def broken(*a, **k):
        raise RuntimeError("bug near secret-town:4")
    monkeypatch.setattr(cc, "paired_stats", broken)
    with pytest.raises(RuntimeError):                  # never stringified into a cell or a table
        cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30)


def test_compare_refuses_an_output_inside_the_repository(tmp_path):
    for out in (os.path.join(REPO, "docs", "crop-constant-2026-11", "compare.json"), os.path.join(REPO, "c.json")):
        with pytest.raises(SystemExit, match="inside the repository"):
            cc.main(["compare", "--work", str(tmp_path), "--out", out])
        assert not os.path.exists(out)
    with open(os.path.join(REPO, ".gitignore")) as f:
        assert "docs/crop-constant-2026-11/compare.json" in f.read().split()


def test_compare_reads_the_registered_seeds_and_takes_no_seed_option(tmp_path, capsys):
    assert cc.REGISTERED_SEEDS == (42, 43, 44, 45, 46)
    with pytest.raises(SystemExit):
        cc.main(["compare", "--work", str(tmp_path), "--seeds", "42,43,44", "--out", str(tmp_path / "c.json")])
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.8)
    cc.main(["compare", "--work", work, "--types", "CurbRamp", "--n-boot", "20", "--out", str(tmp_path / "c.json")])
    with open(tmp_path / "c.json") as f:
        out = json.load(f)
    assert out["seeds"] == [42, 43, 44, 45, 46] and out["readout"]["outcome"] == "unestimable"
    cc.main(["registered-seeds"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == "42,43,44,45,46"


def test_fewer_than_ten_of_a_class_is_unestimable(tmp_path):
    work = str(tmp_path)
    make_family(work, n_pano=100)
    for c in cc.ARMS:                                                  # keep 9 incorrect test rows
        p = os.path.join(cc.arm_dir(work, c), "CurbRamp", "split.csv")
        s = pd.read_csv(p)
        inc = s.index[(s.split == "test") & (s.label_class == "incorrect")][9:]
        s.loc[inc, "split"] = "train"
        s.to_csv(p, index=False)
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44, 45, 46), n_boot=50)
    assert "fewer than 10" in res["cells"][4000]["unestimable"]


def test_compare_reports_slices_with_training_counts_and_the_negative_control(tmp_path):
    work = str(tmp_path / "w")
    split = make_family(work, seeds=(42, 43, 44), gain=0.5)
    ctrl = tmp_path / "r3" / "CurbRamp"
    ctrl.mkdir(parents=True)
    src = os.path.join(cc.arm_dir(work, 6100), "CurbRamp")
    shutil.copy(os.path.join(src, "predictions_s42.csv"), ctrl / "predictions.csv")
    for s in (43, 44):
        shutil.copy(os.path.join(src, f"predictions_s{s}.csv"), ctrl / f"predictions_baseline-s{s}.csv")
    res = cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=50, control_root=str(tmp_path / "r3"))
    ctl = res["negative_control"]
    assert ctl["point"] == 0 and ctl["seeds"] == [42, 43, 44]
    assert ctl["inputs_sha256"]["s42"] == cc.sha256_file(str(ctrl / "predictions.csv"))
    tags = [r for r in res["slices"] if r["by"] == "tag" and r["split"] == "test" and r["arm"] == 4000]
    assert tags and all("train_n" in r and "needs_data" in r and "base_auc" in r for r in tags)
    assert {r["by"] for r in res["slices"]} >= {"severity", "tag", "crop_half_px_quartile", "capture_year"}
    assert set(res["recall_at_precision"][4000]) == {"correct", "incorrect"}
    assert {r["left_out"] for r in res["lodo"][4000]} == {"held-x", "held-y", "secret-held"}
    ctx = cc.context_auc(work, "CurbRamp", os.path.join(src, "predictions_s42.csv"))
    assert ctx["auc"] == pytest.approx(res["auc_test"][6100]["s42"])


# ---------------------------------------------------------------------------------------------------------------------
# report: no non-public deployment is named
# ---------------------------------------------------------------------------------------------------------------------

def test_report_names_no_non_public_deployment(tmp_path):
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.5)
    res = {"CurbRamp": cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30)}
    out = cc.compare_output(res, seeds=(42, 43, 44))
    path = tmp_path / "compare.json"
    cc.write_json(str(path), out)
    pub = tmp_path / "public.json"
    pub.write_text(json.dumps(["pub-a", "pub-b", "held-x", "held-y"]))
    assets = tmp_path / "assets"
    cc.main(["report", "--compare", str(path), "--public-cities", str(pub), "--out-dir", str(assets)])
    text = "".join(p.read_text(errors="ignore") for p in assets.iterdir() if p.suffix in (".md", ".json"))
    assert "secret" not in text
    assert "held-x" in text and "non-public deployment" in text
    assert (assets / "tables.md").exists() and (assets / "crop_constant.png").exists()


def _report(tmp_path, res, seeds=(42, 43, 44)):
    path = tmp_path / "compare.json"
    cc.write_json(str(path), cc.compare_output(res, seeds=seeds))
    pub = tmp_path / "public.json"
    pub.write_text(json.dumps(["pub-a", "pub-b", "held-x", "held-y"]))
    assets = tmp_path / "assets"
    cc.main(["report", "--compare", str(path), "--public-cities", str(pub), "--out-dir", str(assets)])
    return (assets / "tables.md").read_text()


@pytest.mark.parametrize("split, city", [("test", "secret-town"), ("test_heldout", "secret-held"),
                                         ("test", "pub-a")])
def test_a_missing_prediction_names_no_row_in_the_report(tmp_path, split, city):
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.5)
    p = os.path.join(cc.arm_dir(work, 4000), "CurbRamp", "predictions_s43.csv")
    d = pd.read_csv(p)
    gone = d[d.label_uid.str.startswith(city) & (d.split == split)].label_uid.iloc[0]
    d[d.label_uid != gone].to_csv(p, index=False)
    res = {"CurbRamp": cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30)}
    why = res["CurbRamp"]["cells"][4000]["unestimable"]
    assert why == f"c4000 {split} seed 43 has no prediction for 1 of {120 if split == 'test' else 72} rows"
    text = _report(tmp_path, res)
    assert "unestimable" in text and gone not in text and "secret" not in text
    assert not any(re.search(rf"{c}:\d", text) for c in ("pub-a", "pub-b", "held-x", "held-y"))


def test_a_negative_control_with_a_missing_row_names_no_row(tmp_path):
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.5)
    ctrl = tmp_path / "r3" / "CurbRamp"
    ctrl.mkdir(parents=True)
    src = os.path.join(cc.arm_dir(work, 6100), "CurbRamp")
    d = pd.read_csv(os.path.join(src, "predictions_s42.csv"))
    gone = d[d.label_uid.str.startswith("secret-town") & (d.split == "test")].label_uid.iloc[0]
    d[d.label_uid != gone].to_csv(ctrl / "predictions.csv", index=False)
    for s in (43, 44):
        shutil.copy(os.path.join(src, f"predictions_s{s}.csv"), ctrl / f"predictions_baseline-s{s}.csv")
    res = {"CurbRamp": cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30,
                                       control_root=str(tmp_path / "r3"))}
    note = res["CurbRamp"]["negative_control"]["note"]
    assert note == "stored round-3 test seed 42 has no prediction for 1 of 120 rows"
    text = _report(tmp_path, res)
    assert gone not in text and "secret" not in text


def test_report_refuses_text_with_a_non_public_id_or_a_label_id(tmp_path):
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.5)
    res = {"CurbRamp": cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30)}
    for leak in ("secret-town", "pub-a:17"):          # as if some future code path put it into a cell's text
        res["CurbRamp"]["cells"][4000] = {"unestimable": f"broken near {leak}", "k": 3}
        d = tmp_path / leak.replace(":", "_")
        d.mkdir()
        with pytest.raises(SystemExit, match="refusing to write the report"):
            _report(d, res)
        assert not (d / "assets" / "tables.md").exists()
    assert cc.report_leaks("| c4000 | 0.01 |", {"secret-town", "pub-a"}, {"pub-a"}) == []
    assert cc.report_leaks("Secret-Town", {"secret-town"}, set())
    assert cc.report_leaks("| held-x | x |", {"held", "x", "held-x"}, {"held-x"}) == [   # whole tokens only
        "a non-public deployment id (1 of 2)"]                                       # 'x' is a whole token here
    assert cc.report_leaks("| held-x | 0.01 |", {"held", "held-x"}, {"held-x"}) == []
    assert cc.report_leaks("near secret-town:4.", {"secret-town"}, set()) == [
        "a non-public deployment id (1 of 1)", "a label id"]
    assert cc.report_leaks("pub-a:17", {"pub-a"}, {"pub-a"}) == ["a label id"]


@pytest.mark.parametrize("cities", ["absent", [], None])
def test_report_refuses_a_compare_record_without_cities(tmp_path, cities):
    work = str(tmp_path / "w")
    make_family(work, seeds=(42, 43, 44), gain=0.5)
    res = {"CurbRamp": cc.compare_type(work, "CurbRamp", seeds=(42, 43, 44), n_boot=30)}
    res["CurbRamp"]["cells"][4000] = {"unestimable": "broken near secret-town:4", "k": 3}
    if cities == "absent":
        res["CurbRamp"].pop("cities")
    else:
        res["CurbRamp"]["cities"] = cities
    with pytest.raises(SystemExit, match="records no cities"):
        _report(tmp_path, res)
    assert not (tmp_path / "assets" / "tables.md").exists()


# ---------------------------------------------------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------------------------------------------------

LAUNCHERS = ["crop_constant_recrop.slurm", "crop_constant_submit.sh"]


def read_launcher(name):
    with open(os.path.join(REPO, "analysis", name)) as f:
        return f.read()


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("name", LAUNCHERS)
def test_launchers_parse(name):
    r = subprocess.run(["bash", "-n", os.path.join(REPO, "analysis", name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_recrop_job_is_preemptible_and_requeues():
    text = read_launcher("crop_constant_recrop.slurm")
    assert "#SBATCH --partition=ckpt-all" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --account=makelab" in text
    assert "dd if=/dev/zero" in text and "bs=1M count=8" in text      # a real write test first
    assert "SMOKE" in text and "--limit 20" in text
    assert os.access(os.path.join(REPO, "analysis", "crop_constant_submit.sh"), os.X_OK)


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_recrop_job_refuses_any_other_working_directory(tmp_path):
    r = subprocess.run(["bash", os.path.join(REPO, "analysis", "crop_constant_recrop.slurm")], cwd=tmp_path,
                       capture_output=True, text=True, env={**os.environ, "USER": "tester", "STEP": "recrop"})
    assert r.returncode == 2 and "refusing" in r.stdout + r.stderr


def _dry_run(tmp_path, **env):
    root = str(tmp_path / "round3")
    if not os.path.exists(root):
        make_round3(root, write_stored=False)
    e = {**os.environ, "DRY_RUN": "1", "USER": "tester", "R3": root, "RAWLABELS": root + "/rawlabels",
         "PYTHON": sys.executable, **env}
    return subprocess.run(["bash", "analysis/crop_constant_submit.sh"], cwd=REPO, env=e, capture_output=True,
                          text=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_submit_dry_run_prints_the_whole_chain(tmp_path):
    r = _dry_run(tmp_path)
    assert r.returncode == 0, r.stderr
    W = "/gscratch/scrubbed/tester/crop-constant-2026-11"
    lines = r.stdout.splitlines()
    sb = [line for line in lines if line.startswith("sbatch")]
    stage = [line for line in sb if "ai_labels_stage.slurm" in line]
    assert len(stage) == 1 and f"DEST={W}/store" in stage[0] and f"RUN={W}" in stage[0]
    assert "--time=24:00:00" in stage[0] and f"--chdir={W}" in stage[0]
    assert any(f"stage-list --root {tmp_path / 'round3'}" in line and f"--out {W}/stage_files.txt" in line
               for line in lines)
    pre = [line for line in sb if "STEP=preflight" in line]
    rec = [line for line in sb if "STEP=recrop" in line]
    gates = [line for line in sb if "STEP=gates" in line]
    assert len(pre) == len(rec) == len(gates) == 1
    assert "--dependency=afterok:dry1" in pre[0] and "--dependency=afterok:dry2" in rec[0]
    assert "--dependency=afterok:dry3" in gates[0]
    n_chunks = cc.chunk_plan(make_round3_frame(tmp_path), cc.max_labels_for("8:00:00")).nunique()
    assert f"--array=0-{n_chunks - 1}" in rec[0] and "--time=8:00:00" in rec[0] and "TIME_LIMIT=8:00:00" in rec[0]
    train = [line for line in sb if "confident_learning.slurm" in line]
    assert len(train) == 3 * 2 * 5                                     # arms x types x registered seeds
    assert {f"SEED={s}," for s in cc.REGISTERED_SEEDS} == {re.search(r"SEED=\d+,", x).group(0) for x in train}
    for line in train:
        assert "--dependency=afterok:dry4" in line and "VARIANT=baseline" in line and "KEEP_CKPT=0" in line
    runs = {c: [line for line in train if f"DATA={W}/c{c}," in line] for c in cc.ARMS}
    for c, ls in runs.items():
        assert len(ls) == 10 and all(f"RUNS={W}/runs/c{c}," in line for line in ls)  # one RUNS directory per arm
    assert all("--kill-on-invalid-dep=yes" in line for line in sb)


def make_round3_frame(tmp_path):
    root = str(tmp_path / "round3")
    return pd.concat([pd.read_csv(os.path.join(root, t, "split.csv"), dtype={"pano_id": str}) for t in TYPES])


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("dest", ["/gscratch/makelab/tester/pano-stage", "/gscratch/scrubbed/tester/other/store",
                                  "/gscratch/scrubbed/tester/crop-constant-2026-11/../x"])
def test_submit_refuses_a_stage_destination_outside_the_working_directory(tmp_path, dest):
    r = _dry_run(tmp_path, DEST=dest)
    assert r.returncode == 2 and "refusing" in r.stderr and "sbatch" not in r.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_submit_without_restage_reads_a_shared_store_and_needs_the_baseline_arm(tmp_path):
    r = _dry_run(tmp_path, RESTAGE="0", PANO_STORE="/gscratch/scrubbed/tester/shared/store")
    assert r.returncode == 0, r.stderr
    assert "ai_labels_stage.slurm" not in r.stdout
    pre = [line for line in r.stdout.splitlines() if "STEP=preflight" in line][0]
    assert "PANO_STORE=/gscratch/scrubbed/tester/shared/store" in pre and "--dependency" not in pre
    r = _dry_run(tmp_path, ARMS="4000 8000")
    assert r.returncode == 2 and "6100" in r.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_submit_takes_no_seed_setting_and_tests_a_real_write_first(tmp_path):
    r = _dry_run(tmp_path, SEEDS="42,43,44")
    assert r.returncode == 2 and "REGISTERED_SEEDS" in r.stderr and "sbatch" not in r.stdout
    text = read_launcher("crop_constant_submit.sh")
    assert "dd if=/dev/zero" in text and "bs=1M count=8" in text
    assert text.index("dd if=/dev/zero") < text.index("sb --chdir")        # before the first submission
    assert 'JOB=$("${SBATCH[@]}" "$@")' in text and 'echo "${SBATCH[*]} $*"' in text   # one argument list


# ---------------------------------------------------------------------------------------------------------------------
# Several stores (decision 3): the study's own restage first, then a shared store it only reads
# ---------------------------------------------------------------------------------------------------------------------

def _named_store(base, name, files=(), cities=(), marker=True):
    """A store at ``<base>/<name>`` holding ``files`` (``(city, pano_id)``) and an empty directory per ``cities``."""
    store = os.path.join(str(base), name)
    os.makedirs(store, exist_ok=True)
    if marker:
        with open(os.path.join(store, dg.PANO_STORE_MARKER), "w") as f:
            f.write(f"{name}\n")
    for c in cities:
        os.makedirs(os.path.join(store, c), exist_ok=True)
    for c, p in files:
        path = dg.pano_store_path(store, c, p)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"jpg")
    return store


def _listing(*stores):
    return {s: sorted((os.path.relpath(os.path.join(d, n), s), os.path.getmtime(os.path.join(d, n)))
                      for d, _, ns in os.walk(s) for n in ns) for s in stores}


def test_several_stores_read_each_panorama_from_the_first_store_that_holds_it(r3, tmp_path):
    root, frame = r3
    panos = frame[["city", "pano_id"]].drop_duplicates()
    every = list(panos.itertuples(index=False, name=None))
    own_only = [x for x in every if x[1] in ("AApano1", "BBpano3")]
    both = [x for x in every if x[1] == "CCpano5"]
    shared_only = [x for x in every if x[1] not in ("AApano1", "BBpano3", "CCpano5")]
    own = _named_store(tmp_path, "own", own_only + both)
    shared = _named_store(tmp_path, "shared", shared_only + both)
    before = _listing(own, shared)
    work = str(tmp_path / "w")
    _, reader = run_recrop(root, frame, work, pano_store=f"{own}:{shared}")
    read_from = {p: s for s, _, p in reader.calls}
    assert len(reader.calls) == len(read_from) == len(every)            # still one read per panorama
    assert {p for p, s in read_from.items() if s == own} == {"AApano1", "BBpano3", "CCpano5"}   # first store wins
    assert {p for p, s in read_from.items() if s == shared} == {p for _, p in shared_only}
    rec = cc.load_recrop(work).merge(frame[["label_uid", "pano_id"]], on="label_uid", validate="one_to_one")
    assert (rec.read_store == rec.pano_id.map(read_from)).all()          # the store actually read, per row
    assert set(rec.store_root) == {"/stores/r3/CurbRamp", "/stores/r3/SurfaceProblem"}   # round 3's, unchanged
    assert _listing(own, shared) == before                               # nothing is written into any store


def test_a_panorama_missing_from_every_store_reads_no_pano(r3, tmp_path):
    """The real reader: a file in no store is ``no_pano``, read where its city has a directory (here only the second
    store has one), so a store that holds none of a city's panoramas is not mistaken for an outage."""
    root, frame = r3
    target = frame[frame.pano_id == "DDpano7"]
    own = _named_store(tmp_path, "own", cities=["secret-town"])
    shared = _named_store(tmp_path, "shared", cities=["pub-a"])
    work = str(tmp_path / "w")
    cc.recrop(target, work, 0, 1000, pano_store=f"{own}:{shared}", stored_root=root, device=None,
              pano_reader=dg.read_store_panorama, render=FakeRender())
    rec = cc.load_recrop(work)
    assert len(rec) == len(target) and set(rec.status) == {"no_pano"}
    assert set(rec.read_store) == {shared}
    os.remove(os.path.join(shared, dg.PANO_STORE_MARKER))               # a store that went away is never no_pano
    with pytest.raises(dg.PanoStoreUnavailable):
        cc.recrop(target, str(tmp_path / "w2"), 0, 1000, pano_store=f"{own}:{shared}", stored_root=root,
                  device=None, pano_reader=dg.read_store_panorama, render=FakeRender())


def test_one_store_records_the_store_it_read(r3, tmp_path):
    root, frame = r3
    work = str(tmp_path / "w")
    _, reader = run_recrop(root, frame, work, pano_store="/scrubbed/store")
    assert {s for s, _, _ in reader.calls} == {"/scrubbed/store"}
    rec = cc.load_recrop(work)
    assert set(rec.read_store[rec.status.isin(["ok", "native_mismatch", "no_pano"])]) == {"/scrubbed/store"}
    g = cc.gate_g3(rec)
    assert set(g["by_read_store"]) == {"/scrubbed/store"}               # G3 per store actually read


def test_preflight_checks_the_marker_of_every_store(r3_400, tmp_path):
    root, panos = r3_400
    own = _named_store(tmp_path, "own", panos[:100])
    shared = _named_store(tmp_path, "shared", panos[100:])
    rec = _preflight(root, f"{own}:{shared}")
    assert rec["store"]["passed"] and rec["store"]["missing_panoramas"] == 0
    assert rec["store"]["pano_stores"] == [own, shared]
    assert rec["store"]["markers"] == {own: "own", shared: "shared"}
    assert rec["store"]["panoramas_by_store"] == {own: 100, shared: 300}
    os.remove(os.path.join(shared, dg.PANO_STORE_MARKER))
    bad = _preflight(root, f"{own}:{shared}")
    assert not bad["store"]["passed"]
    assert len(bad["store"]["failures"]) == 1 and shared in bad["store"]["failures"][0]
    assert own not in bad["store"]["failures"][0]


def test_the_preflight_command_refuses_a_store_without_its_marker_among_several(r3_400, tmp_path):
    root, panos = r3_400
    own = _named_store(tmp_path, "own", panos[:100])
    shared = _named_store(tmp_path, "shared", panos[100:], marker=False)
    out = str(tmp_path / "preflight.json")
    with pytest.raises(SystemExit) as e:
        cc.main(["preflight", "--root", root, "--rawlabels-dir", os.path.join(root, "rawlabels"), "--types",
                 "CurbRamp", "--pano-store", f"{own}:{shared}", "--out", out])
    assert "preflight failed" in str(e.value.code)
    with open(out) as f:
        store = json.load(f)["store"]
    assert store["pano_stores"] == [own, shared] and shared in store["failures"][0]


def test_stage_list_exclude_store_leaves_out_what_is_already_staged(r3, tmp_path):
    root, frame = r3
    files = cc.stage_files(frame)
    every = list(frame[["city", "pano_id"]].drop_duplicates().itertuples(index=False, name=None))
    own = _named_store(tmp_path, "own", every[:1])
    shared = _named_store(tmp_path, "shared", every[1:3] + every[:1])
    before = _listing(own, shared)
    out = str(tmp_path / "list" / "stage_files.txt")
    cc.main(["stage-list", "--root", root, "--types", ",".join(TYPES), "--out", out, "--exclude-store", own,
             "--exclude-store", shared])
    with open(out) as f:
        listed = f.read().split()
    present = {f"{c}/{p[:2]}/{p}.jpg" for c, p in every[:3]}
    assert listed == [f for f in files if f not in present] and len(listed) == len(files) - 3
    with open(os.path.join(tmp_path, "list", "stage_files.json")) as f:
        rec = json.load(f)
    assert rec["panoramas"] == len(files) and rec["already_in_store"] == 3 and rec["to_transfer"] == len(files) - 3
    assert rec["exclude_stores"] == [own, shared] and rec["already_by_store"] == {own: 1, shared: 2}
    assert rec["stage_files_sha256"] == cc.sha256_file(out)
    assert _listing(own, shared) == before
    cc.main(["stage-list", "--root", root, "--types", ",".join(TYPES), "--out", out, "--exclude-store",
             f"{own}:{shared}"])                                         # a colon-separated list is the same
    with open(out) as f:
        assert f.read().split() == listed


def test_a_chunk_file_from_before_the_read_store_column_is_refused(r3, tmp_path):
    """Appending rows with a new column to a chunk written without it would shift every later column."""
    root, frame = r3
    work = str(tmp_path / "w")
    old = [f for f in cc.RECROP_FIELDS if f != "read_store"]
    os.makedirs(os.path.join(work, "recrop"))
    pd.DataFrame([{"label_uid": frame.label_uid.iloc[0], "status": "error:X"}], columns=old).to_csv(
        cc.chunk_path(work, 0), index=False)
    with pytest.raises(ValueError, match="columns"):
        run_recrop(root, frame, work)


SHARED = "/gscratch/scrubbed/tester/pano-stage/context-arms"


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_submit_dry_run_with_a_shared_store(tmp_path):
    W = "/gscratch/scrubbed/tester/crop-constant-2026-11"
    r = _dry_run(tmp_path, SHARED_STORE=SHARED)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    sl_ = [line for line in lines if " stage-list " in line]
    assert len(sl_) == 1 and f"--exclude-store {W}/store" in sl_[0] and f"--exclude-store {SHARED}" in sl_[0]
    sb = [line for line in lines if line.startswith("sbatch")]
    stage = [line for line in sb if "ai_labels_stage.slurm" in line]
    assert len(stage) == 1 and f"DEST={W}/store" in stage[0] and SHARED not in stage[0]   # the pull writes only DEST
    steps = [line for line in sb if "crop_constant_recrop.slurm" in line]
    assert len(steps) == 3 and all(f"PANO_STORE={W}/store:{SHARED}," in line for line in steps)
    train = [line for line in sb if "confident_learning.slurm" in line]
    assert len(train) == 3 * 2 * 5 and not any(SHARED in line for line in train)
    for line in lines:                         # the shared store is only ever an excluded or a read store
        rest = line.replace(f"--exclude-store {SHARED}", "").replace(f"PANO_STORE={W}/store:{SHARED},", "")
        assert SHARED not in rest, line
    explicit = _dry_run(tmp_path, SHARED_STORE=SHARED, PANO_STORE=f"{SHARED}:{W}/store")
    assert explicit.returncode == 0 and f"PANO_STORE={SHARED}:{W}/store," in explicit.stdout   # PANO_STORE still wins


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
@pytest.mark.parametrize("shared", ["/gscratch/scrubbed/tester/crop-constant-2026-11/store2",
                                    "/mmfs1/gscratch/scrubbed/tester/crop-constant-2026-11/x",
                                    "/gscratch/scrubbed/tester/crop-constant-2026-11",
                                    "/gscratch/scrubbed/tester", "/gscratch/scrubbed/tester/",
                                    "relative/store", "/gscratch/scrubbed/tester/a/../crop-constant-2026-11/x",
                                    "/a:/b", "/a,b"])
def test_submit_refuses_a_shared_store_the_study_would_write_into(tmp_path, shared):
    r = _dry_run(tmp_path, SHARED_STORE=shared)
    assert r.returncode == 2 and "refusing" in r.stderr and "SHARED_STORE" in r.stderr and "sbatch" not in r.stdout
