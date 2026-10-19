"""data_generation.main() end to end on a CPU with the depth model stubbed and panoramas read from a tiny synthetic
pano store, plus the store reader itself: what the manifest records, resume guards, and failure handling."""
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import data_generation as dg
import split_data

AI = dg.AI_USER_ID
DEPTH_AT_CENTRE = 7.3
CITY = "toy-city"
PANO_BGR = (200, 100, 30)  # a solid pano in OpenCV channel order, so a BGR/RGB swap is visible in the crop


def _votes(agree, disagree):
    return json.dumps([{"validation": "Agree", "validator_type": "Human"}] * agree
                      + [{"validation": "Disagree", "validator_type": "Human"}] * disagree)


def write_city(tmp_path, rows):
    """The city's rawLabels export and a one-city /v3/api/cities listing in the cache, so main() needs no network."""
    pd.DataFrame(rows).to_csv(tmp_path / f"{CITY}.csv", index=False)
    (tmp_path / dg.CITIES_CACHE_FILE).write_text(json.dumps({"fetched_on": "2026-10-19", "url": dg.CITIES_API,
        "cities": [{"city_id": CITY, "visibility": "public", "url": "https://sidewalk-toy.cs.washington.edu"}]}))


def label(i, agree, disagree, user="u", pano=None):
    return {"label_id": i, "user_id": user, "pano_id": pano or f"p{i}", "pano_source": "gsv",
            "label_type": "CurbRamp", "validations": _votes(agree, disagree), "pano_x": 3000 + 10 * i,
            "pano_y": 2600, "pano_width": 8192, "pano_height": 4096, "camera_pitch": 2.0, "camera_roll": None,
            "severity": 1, "tags": "[]", "high_quality_user": True, "image_capture_date": "2024-09"}


def write_pano(store, city, pano_id, size=(64, 32), bgr=PANO_BGR):
    """A solid ``size`` = (w, h) jpg at the store path the API and the generator both read."""
    path = store / city / pano_id[:2] / f"{pano_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((size[1], size[0], 3), bgr, np.uint8))
    return path


def mark_store(store, contents="toy store v1\nsecond line\n"):
    store.mkdir(parents=True, exist_ok=True)
    (store / dg.PANO_STORE_MARKER).write_text(contents)
    return store


def make_store(tmp_path, rows, city=CITY):
    store = mark_store(tmp_path / "store")
    (store / city).mkdir(parents=True, exist_ok=True)  # the city directory exists even if every pano is removed
    for pano_id in {r["pano_id"] for r in rows}:
        write_pano(store, city, pano_id)
    return store


def remove_panos(store, pano_ids, city=CITY):
    for p in pano_ids:
        os.remove(dg.pano_store_path(str(store), city, p))


def test_vote_thresholds_at_the_boundaries(tmp_path):
    rows = [label(1, 3, 0), label(2, 2, 0), label(3, 0, 2), label(4, 0, 1)]
    write_city(tmp_path, rows)
    eligible, _ = dg.eligible_labels(pd.read_csv(tmp_path / f"{CITY}.csv").assign(city=CITY), "CurbRamp")
    assert dict(zip(eligible.label_id, eligible.label_class)) == {1: "correct", 3: "incorrect"}


class FakeDepth:
    def __init__(self):
        self.calls = 0
        self.centre_bgr = None

    def infer_image(self, bgr):
        self.calls += 1
        assert bgr.shape == (dg.VIEW_SIZE, dg.VIEW_SIZE, 3)
        self.centre_bgr = tuple(int(v) for v in bgr[dg.VIEW_SIZE // 2, dg.VIEW_SIZE // 2])
        d = np.full(bgr.shape[:2], 20.0, np.float32)
        d[dg.VIEW_SIZE // 2, dg.VIEW_SIZE // 2] = DEPTH_AT_CENTRE
        return d


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Two human panos (one shared by two labels) and one AI-authored label, all in a synthetic store; depth stubbed.

    Store panos are resized to 2048x1024 here instead of the real 8192x4096 (``PANO_SIZE``), which keeps the numpy
    renderer fast; the real constant is tested on the reader directly."""
    rows = [label(1, 4, 0, pano="shared"), label(2, 0, 3, pano="shared"), label(3, 0, 3),
            label(4, 0, 4, user=AI), label(5, 5, 0)]
    write_city(tmp_path, rows)
    store = make_store(tmp_path, rows)
    monkeypatch.setattr(dg, "PANO_SIZE", (2048, 1024))
    read = []
    real_read = dg.read_store_panorama

    def spy(store_dir, city, pano_id):
        read.append(pano_id)
        return real_read(store_dir, city, pano_id)
    depth = FakeDepth()
    monkeypatch.setattr(dg, "read_store_panorama", spy)
    monkeypatch.setattr(dg, "load_depth_model", lambda encoder, device: depth)
    return {"tmp": tmp_path, "store": store, "read": read, "depth": depth}


def run_main(tmp, *extra, store=None):
    dg.main(["--label_type", "CurbRamp", "--rawlabels-cache", str(tmp), "--out", str(tmp / "out"),
             "--pano-store", str(store or tmp / "store"), "--include-ai-labels", "--allow-cpu", *extra])
    return pd.read_csv(tmp / "out" / "CurbRamp" / "manifest.csv")


def test_main_records_tilt_depth_and_crop_size(stubbed):
    m = run_main(stubbed["tmp"]).set_index("label_uid")
    assert (m.status == "ok").all()
    assert m.loc["toy-city:4", "author"] == "ai" and m.loc["toy-city:4", "tilt_applied"] == 0
    assert (m.loc[m.author == "human", "tilt_applied"] == 1).all()
    assert np.allclose(m.depth_m, DEPTH_AT_CENTRE)
    assert (m.crop_half_px == int(6100 / DEPTH_AT_CENTRE)).all()
    assert stubbed["read"].count("shared") == 1  # consecutive labels on one pano reuse it
    run = json.loads((stubbed["tmp"] / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["settings"] == {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                               "renderer": "numpy", "include_ai_labels": True,
                               "package_source": dg.package_source()}
    assert run["pano_store"] == str(stubbed["store"].resolve())  # provenance, outside the resume settings
    assert run["pano_store_marker"] == "toy store v1"  # the marker's first line


def test_manifest_records_provider_and_store_path(stubbed):
    m = run_main(stubbed["tmp"]).set_index("label_uid")
    assert "pano_zoom" not in m.columns
    assert (m.pano_source == "gsv").all()  # the provider column is unchanged
    assert m.loc["toy-city:3", "pano_path"] == str(stubbed["store"] / CITY / "p3" / "p3.jpg")
    assert (m.pano_native_w == 64).all() and (m.pano_native_h == 32).all()  # the stored size, before the resize


def test_relative_store_path_is_made_absolute_once(stubbed, monkeypatch):
    monkeypatch.chdir(stubbed["tmp"])
    m = run_main(stubbed["tmp"], store="store")  # relative to the working directory
    assert m.pano_path.str.startswith(str(stubbed["store"].resolve())).all()
    run = json.loads((stubbed["tmp"] / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["pano_store"] == str(stubbed["store"].resolve())


def test_store_pixels_reach_the_crop_as_rgb(stubbed):
    """The store jpg is BGR (cv2.imread); depth gets the BGR view, the saved crop is RGB."""
    m = run_main(stubbed["tmp"], "--limit", "1")
    assert np.allclose(stubbed["depth"].centre_bgr, PANO_BGR, atol=3)
    crop = np.asarray(Image.open(stubbed["tmp"] / "out" / "CurbRamp" / m.crop_path[0]).convert("RGB"))
    assert np.allclose(crop[crop.shape[0] // 2, crop.shape[1] // 2], PANO_BGR[::-1], atol=3)


def test_resume_refuses_changed_settings_and_keeps_the_record(stubbed):
    tmp = stubbed["tmp"]
    first = run_main(tmp)
    run_path = tmp / "out" / "CurbRamp" / "generation_run.json"
    before = run_path.read_text()
    with pytest.raises(SystemExit, match=r"tilt_beta 0\.95 -> 0\.5.*holds 5 ok rows"):
        run_main(tmp, "--tilt-beta", "0.5")
    assert run_path.read_text() == before  # a refused resume never rewrites the settings
    calls = stubbed["depth"].calls
    again = run_main(tmp, "--tilt-beta", "0.5", "--force-reselect")
    assert stubbed["depth"].calls == calls + len(first)  # every crop redone under the new beta
    assert (again.tilt_beta == 0.5).all()
    assert json.loads(run_path.read_text())["settings"]["tilt_beta"] == 0.5


def test_moving_the_store_does_not_refuse_a_resume(stubbed):
    """The store path is a location, not a selection setting: the same panos under a new mount resume cleanly."""
    tmp = stubbed["tmp"]
    run_main(tmp)
    moved = tmp / "mnt" / "store"
    moved.parent.mkdir()
    stubbed["store"].rename(moved)
    calls = stubbed["depth"].calls
    run_main(tmp, store=moved)
    assert stubbed["depth"].calls == calls  # every ok row kept
    assert json.loads((tmp / "out" / "CurbRamp" / "generation_run.json").read_text())["pano_store"] == str(moved)


def test_plain_resume_skips_ok_rows(stubbed):
    run_main(stubbed["tmp"])
    calls = stubbed["depth"].calls
    run_main(stubbed["tmp"])
    assert stubbed["depth"].calls == calls


def test_fatal_errors_are_raised_not_recorded(stubbed, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("broken invariant")
    monkeypatch.setattr(dg, "create_perspective_crop", boom)
    with pytest.raises(AssertionError):
        run_main(stubbed["tmp"])
    assert dg.is_fatal(RuntimeError("CUDA error: device-side assert"))
    assert not dg.is_fatal(ValueError("bad depth"))


def test_aborts_after_consecutive_errors(stubbed, monkeypatch):
    def fail(*a, **k):
        raise ValueError("bad depth")
    monkeypatch.setattr(dg, "create_perspective_crop", fail)
    monkeypatch.setattr(dg, "MAX_CONSECUTIVE_ERRORS", 2)
    with pytest.raises(SystemExit, match="2 consecutive"):
        run_main(stubbed["tmp"])


# ---------------------------------------------------------------- the store reader

def test_pano_size_is_the_serving_size():
    assert dg.PANO_SIZE == (8192, 4096)


def test_read_store_panorama_resizes_like_the_api(tmp_path):
    """cv2.imread + resize to 8192x4096 with INTER_LINEAR, the method of sidewalk-ai-api's cache path (byte-identity
    across the two repos also needs the same opencv build)."""
    path = write_pano(tmp_path, "seattle-wa", "abcdef")
    pano, info = dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")
    assert info == {"pano_path": str(path), "pano_native_w": 64, "pano_native_h": 32}
    assert pano.shape == (4096, 8192, 3)
    expected = cv2.resize(cv2.imread(str(path)), (8192, 4096), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(pano, expected)


def test_missing_file_is_no_pano(tmp_path):
    mark_store(tmp_path)
    (tmp_path / "seattle-wa").mkdir()
    assert dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef") == (
        "no_pano", {"pano_path": None, "pano_native_w": None, "pano_native_h": None})


def test_missing_file_with_the_store_gone_is_fatal(tmp_path):
    """A no_pano is only believed while the marker and the city directory are still there."""
    (tmp_path / "seattle-wa").mkdir()  # no marker: e.g. the mount dropped after the startup check
    with pytest.raises(dg.PanoStoreUnavailable, match="marker"):
        dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")
    mark_store(tmp_path)
    with pytest.raises(dg.PanoStoreUnavailable, match="city directory"):
        dg.read_store_panorama(str(tmp_path), "cdmx", "abcdef")
    assert dg.is_fatal(dg.PanoStoreUnavailable("x"))


def test_io_error_other_than_missing_is_fatal(tmp_path, monkeypatch):
    """A stale NFS handle or a dropped mount (errno 107 here) must stop the run, not become no_pano."""
    write_pano(tmp_path, "seattle-wa", "abcdef")
    mark_store(tmp_path)
    real_stat = dg.os.stat

    def flaky_stat(path, *a, **k):
        if str(path).endswith(".jpg"):
            raise OSError(107, "Transport endpoint is not connected", str(path))
        return real_stat(path, *a, **k)
    monkeypatch.setattr(dg.os, "stat", flaky_stat)
    with pytest.raises(dg.PanoStoreUnavailable, match="not connected"):
        dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")


def test_unreadable_file_is_bad_pano(tmp_path):
    (tmp_path / dg.PANO_STORE_MARKER).write_text("store\n")
    path = tmp_path / "seattle-wa" / "ab" / "abcdef.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a jpeg at all")
    assert dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef") == (
        "bad_pano", {"pano_path": str(path), "pano_native_w": None, "pano_native_h": None})


def test_unreadable_file_on_a_dropped_store_is_fatal(tmp_path):
    """A mount that drops between the stat and the read makes imread fail; that is an outage, not a bad file."""
    path = tmp_path / "seattle-wa" / "ab" / "abcdef.jpg"   # no marker: the store is gone
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a jpeg at all")
    with pytest.raises(dg.PanoStoreUnavailable, match="marker"):
        dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")


def test_mid_run_marker_must_be_a_file(tmp_path):
    """The mid-run re-check holds the marker to the same standard as startup: a directory of that name is no marker."""
    (tmp_path / dg.PANO_STORE_MARKER).mkdir()
    (tmp_path / "seattle-wa").mkdir()
    with pytest.raises(dg.PanoStoreUnavailable, match="wrong kind"):
        dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")


def test_non_2_to_1_file_is_bad_pano(tmp_path):
    path = write_pano(tmp_path, "seattle-wa", "abcdef", size=(64, 48))  # e.g. a cropped or truncated download
    assert dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef") == (
        "bad_pano", {"pano_path": str(path), "pano_native_w": 64, "pano_native_h": 48})


def test_city_key_is_part_of_the_path(tmp_path):
    """The same pano_id in two cities is two files; each label reads its own city's."""
    write_pano(tmp_path, "seattle-wa", "abcdef", bgr=(10, 10, 10))
    write_pano(tmp_path, "teaneck-nj", "abcdef", bgr=(240, 240, 240))
    a, ia = dg.read_store_panorama(str(tmp_path), "seattle-wa", "abcdef")
    b, ib = dg.read_store_panorama(str(tmp_path), "teaneck-nj", "abcdef")
    assert ia["pano_path"].endswith("seattle-wa/ab/abcdef.jpg") and ib["pano_path"].endswith("teaneck-nj/ab/abcdef.jpg")
    assert a.mean() < 50 < 200 < b.mean()


def test_missing_store_fails_fast(stubbed, monkeypatch):
    def no_download(*a, **k):
        raise AssertionError("labels were loaded before the store was checked")
    monkeypatch.setattr(dg, "load_city", no_download)
    with pytest.raises(SystemExit, match="is not a directory"):
        run_main(stubbed["tmp"], store=stubbed["tmp"] / "nope")
    unmarked = stubbed["tmp"] / "unmarked"
    (unmarked / CITY).mkdir(parents=True)
    with pytest.raises(SystemExit, match=r"no \.pano-store marker"):
        run_main(stubbed["tmp"], store=unmarked)
    empty = mark_store(stubbed["tmp"] / "empty-store")
    (empty / "other-city").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no directory for any selected city"):
        run_main(stubbed["tmp"], store=empty)
    with pytest.raises(SystemExit):  # argparse error: --pano-store is required unless --select-only
        dg.main(["--label_type", "CurbRamp", "--rawlabels-cache", str(stubbed["tmp"]), "--allow-cpu"])


def test_empty_marker_is_accepted(stubbed):
    mark_store(stubbed["store"], contents="")
    run_main(stubbed["tmp"], "--limit", "1")
    run = json.loads((stubbed["tmp"] / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["pano_store_marker"] == ""


def _second_city(stubbed):
    """A selected city whose labels are in the rawLabels cache but which has no store directory."""
    pd.DataFrame([label(10, 0, 3, pano="q10")]).to_csv(stubbed["tmp"] / "other-city.csv", index=False)
    return ["--cities", f"{CITY},other-city"]


def test_absent_city_directory_is_fatal_even_on_a_limit_run(stubbed):
    for extra in ([], ["--limit", "2"]):
        with pytest.raises(SystemExit, match="no directory for other-city.*--allow-missing-cities"):
            run_main(stubbed["tmp"], *_second_city(stubbed), *extra)


def test_allow_missing_cities_makes_their_labels_no_pano(stubbed):
    m = run_main(stubbed["tmp"], *_second_city(stubbed), "--allow-missing-cities").set_index("label_uid")
    assert m.loc["other-city:10", "status"] == "no_pano"
    assert (m.drop(index="other-city:10").status == "ok").all()


def test_marker_gone_mid_run_is_fatal(stubbed, monkeypatch):
    """The startup check passes, then the store goes away: the first no_pano is checked against the marker and stops
    the run instead of writing a row."""
    remove_panos(stubbed["store"], ["p4"])  # read order p3, p4, p5, shared
    real_check = dg.check_pano_store

    def check_then_unmount(*a, **k):
        result = real_check(*a, **k)
        (stubbed["store"] / dg.PANO_STORE_MARKER).unlink()
        return result
    monkeypatch.setattr(dg, "check_pano_store", check_then_unmount)
    with pytest.raises(dg.PanoStoreUnavailable, match="marker"):
        run_main(stubbed["tmp"])
    m = pd.read_csv(stubbed["tmp"] / "out" / "CurbRamp" / "manifest.csv")
    assert "no_pano" not in set(m.status)


def test_dropped_mount_mid_run_is_fatal_not_no_pano(stubbed, monkeypatch):
    real_stat = dg.os.stat

    def dropped(path, *a, **k):
        if str(path).endswith("p4.jpg"):
            raise OSError(107, "Transport endpoint is not connected", str(path))
        return real_stat(path, *a, **k)
    monkeypatch.setattr(dg.os, "stat", dropped)
    with pytest.raises(dg.PanoStoreUnavailable):
        run_main(stubbed["tmp"])
    m = pd.read_csv(stubbed["tmp"] / "out" / "CurbRamp" / "manifest.csv")
    assert set(m.status) == {"ok"}  # p3 only; no row for p4


# ---------------------------------------------------------------- missing and unusable panoramas

def test_too_many_missing_panos_fails_the_run(stubbed):
    remove_panos(stubbed["store"], ["p4", "shared"])  # 3 of 5 labels' panos
    with pytest.raises(SystemExit, match="above --max-missing-frac 0.2"):
        run_main(stubbed["tmp"])
    run_main(stubbed["tmp"], "--max-missing-frac", "0.9")  # the same shrinkage, accepted explicitly


def test_consecutive_missing_panos_abort(stubbed, monkeypatch):
    remove_panos(stubbed["store"], ["p3", "p4", "p5", "shared"])  # the city directory stays, empty
    monkeypatch.setattr(dg, "MAX_CONSECUTIVE_MISSING_PANOS", 3)
    with pytest.raises(SystemExit, match=r"3 panoramas in a row.*Check the --pano-store path.*city directory names"):
        run_main(stubbed["tmp"])


def test_bad_pano_is_a_status(stubbed):
    for p in ("p3", "p4", "p5", "shared"):
        with open(dg.pano_store_path(str(stubbed["store"]), CITY, p), "wb") as f:
            f.write(b"\xff\xd8 truncated")
    with pytest.raises(SystemExit, match="no_pano/bad_pano"):  # all missing -> non-zero exit
        run_main(stubbed["tmp"])
    m = pd.read_csv(stubbed["tmp"] / "out" / "CurbRamp" / "manifest.csv")
    assert set(m.status) == {"bad_pano"} and m.pano_path.str.endswith(".jpg").all()


def test_legacy_manifest_is_redone_from_the_store(stubbed, monkeypatch, capsys):
    """A manifest from the tile fetcher (pano_zoom, no pano_path): its ok rows are stale and redone from the store,
    its incomplete_pano rows are retried, and a pano still missing is not a first-time failure for the
    consecutive-missing stop."""
    tmp = stubbed["tmp"]
    run_main(tmp)
    path = tmp / "out" / "CurbRamp" / "manifest.csv"
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    old.loc[old.pano_id.isin(["p3", "p4", "p5"]), "status"] = "incomplete_pano"
    old = old.drop(columns=["pano_path", "pano_native_w", "pano_native_h"]).assign(pano_zoom="4")
    old.to_csv(path, index=False)  # the old column set; the "shared" pano's two rows stay ok
    remove_panos(stubbed["store"], ["p3", "p4", "p5"])
    monkeypatch.setattr(dg, "MAX_CONSECUTIVE_MISSING_PANOS", 2)
    calls = stubbed["depth"].calls
    capsys.readouterr()
    m = run_main(tmp, "--max-missing-frac", "0.9")
    out = capsys.readouterr().out
    assert "Redoing 2 ok rows made by the removed tile fetcher" in out and "predate the pano store" in out
    assert stubbed["depth"].calls == calls + 2  # both pre-store crops re-rendered
    assert list(m.columns) == dg.MANIFEST_COLUMNS
    assert sorted(m.status) == ["no_pano"] * 3 + ["ok"] * 2
    assert m.loc[m.status == "ok", "pano_path"].str.endswith("shared.jpg").all()


# ---------------------------------------------------------------- split order

def test_assign_splits_order_is_deterministic():
    rng = np.random.default_rng(0)
    m = pd.DataFrame({"label_uid": [f"c:{i}" for i in range(40)], "city": "c",
                      "pano_id": [f"p{i // 2}" for i in range(40)],
                      "label_class": rng.choice(["correct", "incorrect"], 40), "status": "ok"})
    a = split_data.assign_splits(m, seed=5)
    b = split_data.assign_splits(m.sample(frac=1, random_state=1), seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert a.label_uid.tolist() == sorted(a.label_uid)


# ---------------------------------------------------------------- resume regressions (verification of 8975153)

def test_resume_after_one_missing_pano_passes(stubbed):
    """1 of 5 labels no_pano (20%, not above 0.2): the first run and a plain resume both exit 0. The resume retries
    only that label, which fails again, and is judged on the cumulative manifest (still 1 of 5)."""
    remove_panos(stubbed["store"], ["p3"])
    m = run_main(stubbed["tmp"])
    assert (m.status == "no_pano").sum() == 1 and len(m) == 5
    m = run_main(stubbed["tmp"])
    assert (m.status == "no_pano").sum() == 1 and len(m) == 5


def test_completed_run_rerun_has_nothing_to_do(stubbed, capsys):
    run_main(stubbed["tmp"])
    capsys.readouterr()
    run_main(stubbed["tmp"])
    assert "Nothing to do" in capsys.readouterr().out


def test_resume_does_not_count_previously_missing_panos_as_consecutive(tmp_path, monkeypatch):
    """Four missing panos among 20 with the consecutive limit at 3: the first run passes (they are spread out), and
    so does the resume, which reads exactly those four back to back."""
    rows = ([label(i, 0, 3, pano=f"p{i:02d}") for i in range(1, 11)]
            + [label(i, 5, 0, pano=f"p{i:02d}") for i in range(11, 21)])
    write_city(tmp_path, rows)
    store = make_store(tmp_path, rows)
    remove_panos(store, ["p02", "p05", "p08", "p12"])
    monkeypatch.setattr(dg, "PANO_SIZE", (2048, 1024))
    monkeypatch.setattr(dg, "load_depth_model", lambda e, d: FakeDepth())
    monkeypatch.setattr(dg, "MAX_CONSECUTIVE_MISSING_PANOS", 3)
    m = run_main(tmp_path, "--max-missing-frac", "0.9")
    assert len(m) == 20 and (m.status == "no_pano").sum() == 4
    m = run_main(tmp_path, "--max-missing-frac", "0.9")
    assert len(m) == 20 and (m.status == "no_pano").sum() == 4


def test_changed_package_commit_refuses_resume(stubbed, monkeypatch):
    run_main(stubbed["tmp"])
    monkeypatch.setattr(dg, "package_source", lambda dist="sidewalk-panorama-tools": "https://example@deadbeef")
    with pytest.raises(SystemExit, match="package_source"):
        run_main(stubbed["tmp"])
    run_main(stubbed["tmp"], "--force-reselect")


def test_changed_city_list_refuses_resume_and_keeps_crops(stubbed, capsys):
    """A city joining the listing stops a resume even when it adds no selected label (the selection hash is
    unchanged); --force-reselect then keeps every crop, unlike a settings change."""
    tmp = stubbed["tmp"]
    run_main(tmp)
    pd.DataFrame([{**label(1, 5, 0), "label_type": "Obstacle"}]).to_csv(tmp / "new-city.csv", index=False)
    (stubbed["store"] / "new-city").mkdir()
    listing = json.loads((tmp / "cities.json").read_text())
    listing["cities"].append({"city_id": "new-city", "visibility": "public",
                              "url": "https://sidewalk-new.cs.washington.edu"})
    (tmp / "cities.json").write_text(json.dumps(listing))
    with pytest.raises(SystemExit, match=r"city list changed \(added: new-city; removed: none\)"):
        run_main(tmp)
    run = json.loads((tmp / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["cities"] == [CITY]  # left untouched by the refusal
    capsys.readouterr()
    m = run_main(tmp, "--force-reselect")
    assert (m.status == "ok").all() and "skipped (already ok)': 5" in capsys.readouterr().out
    run = json.loads((tmp / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["cities"] == ["new-city", CITY] and run["city_source"]["from"] == "cache"


def test_summary_counts_non_gsv_labels(tmp_path, monkeypatch, capsys):
    """Non-GSV labels stay out of the selection (docs/decisions.md) and the end-of-run summary says how many."""
    rows = [label(1, 4, 0), label(2, 0, 3), {**label(3, 0, 3), "pano_source": "panoramax"}]
    write_city(tmp_path, rows)
    make_store(tmp_path, rows)
    monkeypatch.setattr(dg, "PANO_SIZE", (2048, 1024))
    monkeypatch.setattr(dg, "load_depth_model", lambda e, d: FakeDepth())
    m = run_main(tmp_path)
    assert "toy-city:3" not in set(m.label_uid)
    summary = capsys.readouterr().out.split("Processing complete.")[1]
    assert "dropped for unusable pano geometry (before selection): {'non_gsv': 1}" in summary


def test_priors_are_not_written_on_a_refused_resume(stubbed):
    tmp = stubbed["tmp"]
    run_main(tmp)
    priors = tmp / "out" / "deployment_priors.json"
    assert priors.exists()
    priors.unlink()
    with pytest.raises(SystemExit, match="Refusing to resume"):
        run_main(tmp, "--tilt-beta", "0.5")
    assert not priors.exists()
