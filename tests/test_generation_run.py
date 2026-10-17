"""data_generation.main() end to end on a CPU with the panorama fetcher and the depth model stubbed, plus the tile
fetcher against a stubbed tile server: what the manifest records, resume guards, and failure handling."""
import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import data_generation as dg
import split_data

REAL_FETCH = dg.fetch_panorama

AI = dg.AI_USER_ID
DEPTH_AT_CENTRE = 7.3


def _votes(agree, disagree):
    return json.dumps([{"validation": "Agree", "validator_type": "Human"}] * agree
                      + [{"validation": "Disagree", "validator_type": "Human"}] * disagree)


def write_city(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "toy-city.csv", index=False)


def label(i, agree, disagree, user="u", pano=None):
    return {"label_id": i, "user_id": user, "pano_id": pano or f"p{i}", "pano_source": "gsv",
            "label_type": "CurbRamp", "validations": _votes(agree, disagree), "pano_x": 3000 + 10 * i,
            "pano_y": 2600, "pano_width": 8192, "pano_height": 4096, "camera_pitch": 2.0, "camera_roll": None,
            "severity": 1, "tags": "[]", "high_quality_user": True, "image_capture_date": "2024-09"}


def test_vote_thresholds_at_the_boundaries(tmp_path):
    rows = [label(1, 3, 0), label(2, 2, 0), label(3, 0, 2), label(4, 0, 1)]
    write_city(tmp_path, rows)
    eligible, _ = dg.eligible_labels(pd.read_csv(tmp_path / "toy-city.csv").assign(city="toy-city"), "CurbRamp")
    assert dict(zip(eligible.label_id, eligible.label_class)) == {1: "correct", 3: "incorrect"}


class FakeDepth:
    def __init__(self):
        self.calls = 0

    def infer_image(self, bgr):
        self.calls += 1
        assert bgr.shape == (dg.VIEW_SIZE, dg.VIEW_SIZE, 3)
        d = np.full(bgr.shape[:2], 20.0, np.float32)
        d[dg.VIEW_SIZE // 2, dg.VIEW_SIZE // 2] = DEPTH_AT_CENTRE
        return d


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Two human panos (one shared by two labels) and one AI-authored label; fetcher and depth model stubbed."""
    rows = [label(1, 4, 0, pano="shared"), label(2, 0, 3, pano="shared"), label(3, 0, 3),
            label(4, 0, 4, user=AI), label(5, 5, 0)]
    write_city(tmp_path, rows)
    monkeypatch.setattr(dg, "CITIES", {"toy-city": "toy"})
    fetched = []

    def fake_fetch(pano_id, info=None):
        fetched.append(pano_id)
        return np.full((1024, 2048, 3), 90, np.uint8)
    depth = FakeDepth()
    monkeypatch.setattr(dg, "fetch_panorama", fake_fetch)
    monkeypatch.setattr(dg, "load_depth_model", lambda encoder, device: depth)
    return {"tmp": tmp_path, "fetched": fetched, "depth": depth}


def run_main(tmp, *extra):
    dg.main(["--label_type", "CurbRamp", "--rawlabels-cache", str(tmp), "--out", str(tmp / "out"),
             "--include-ai-labels", "--allow-cpu", *extra])
    return pd.read_csv(tmp / "out" / "CurbRamp" / "manifest.csv")


def test_main_records_tilt_depth_and_crop_size(stubbed):
    m = run_main(stubbed["tmp"]).set_index("label_uid")
    assert (m.status == "ok").all()
    assert m.loc["toy-city:4", "author"] == "ai" and m.loc["toy-city:4", "tilt_applied"] == 0
    assert (m.loc[m.author == "human", "tilt_applied"] == 1).all()
    assert np.allclose(m.depth_m, DEPTH_AT_CENTRE)
    assert (m.crop_half_px == int(6100 / DEPTH_AT_CENTRE)).all()
    assert stubbed["fetched"].count("shared") == 1  # consecutive labels on one pano reuse it
    run = json.loads((stubbed["tmp"] / "out" / "CurbRamp" / "generation_run.json").read_text())
    assert run["settings"] == {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti",
                               "renderer": "numpy", "include_ai_labels": True,
                               "package_source": dg.package_source()}


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


# ---------------------------------------------------------------- tile fetcher against a stubbed tile server

def _tile_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


def tile_server(drop=(), fail_first=None):
    """zoom-4 GSV-like grid 16 x 8: real tiles inside, black tiles outside, 404 for ``drop``; ``fail_first`` is
    ``{(x, y): n}``: answer 429 to the first n requests for that tile."""
    real, black = _tile_bytes((120, 80, 40)), _tile_bytes((0, 0, 0))
    fail_first = dict(fail_first or {})

    def get(url):
        q = dict(kv.split("=") for kv in url.split("?")[1].split("&"))
        x, y = int(q["x"]), int(q["y"])
        if (x, y) in drop:
            return FakeResponse(404)
        if fail_first.get((x, y), 0) > 0:
            fail_first[(x, y)] -= 1
            return FakeResponse(429)
        return FakeResponse(200, real if x < 16 and y < 8 else black)
    return get


def test_fetch_panorama_assembles_a_complete_grid(monkeypatch):
    monkeypatch.setattr(dg, "_http_get", tile_server())
    pano = dg.fetch_panorama("p")
    assert pano.shape == (4096, 8192, 3)


def test_fetch_panorama_refuses_an_incomplete_grid(monkeypatch, capsys):
    monkeypatch.setattr(dg, "_http_get", tile_server(drop={(0, 0)}))
    with pytest.raises(dg.IncompletePanorama, match="1 of 128 tiles missing"):
        dg.fetch_panorama("p")
    assert "404" in capsys.readouterr().out  # non-200 responses are reported


def test_probe_retries_a_failed_bottom_row_tile(monkeypatch, capsys):
    """(13, 7) is on the size probe's bottom-row walk. One 429 there used to read as the right edge (a 13-tile-wide
    pano stretched to 8192); now it is retried, and the pano comes out whole at zoom 4."""
    monkeypatch.setattr(dg, "_http_get", tile_server(fail_first={(13, 7): 1}))
    info = {}
    assert dg.fetch_panorama("p", info=info).shape == (4096, 8192, 3) and info["zoom"] == 4
    assert "probe tile (13, 7) failed; retrying once" in capsys.readouterr().out


def test_probe_failure_that_persists_is_incomplete_never_ok(monkeypatch, stubbed):
    """If the probe tile fails twice, the grid comes out 13 x 8 tiles: not 2:1, so incomplete_pano, never ok."""
    monkeypatch.setattr(dg, "_http_get", tile_server(fail_first={(13, 7): 2}))
    with pytest.raises(dg.IncompletePanorama, match="not 2:1"):
        REAL_FETCH("p")
    monkeypatch.setattr(dg, "_http_get", tile_server(fail_first={(13, 7): 2}))
    monkeypatch.setattr(dg, "fetch_panorama", REAL_FETCH)  # the real fetcher against the stub server
    with pytest.raises(SystemExit):
        run_main(stubbed["tmp"], "--limit", "1")
    m = pd.read_csv(stubbed["tmp"] / "out" / "CurbRamp" / "manifest.csv")
    assert m.status.tolist() == ["incomplete_pano"]


def test_too_many_missing_panos_fails_the_run(stubbed, monkeypatch):
    calls = {"n": 0}

    def flaky(pano_id, info=None):  # every other pano missing: 2 of 5 labels' panos
        calls["n"] += 1
        return None if calls["n"] % 2 == 0 else np.full((1024, 2048, 3), 90, np.uint8)
    monkeypatch.setattr(dg, "fetch_panorama", flaky)
    with pytest.raises(SystemExit, match="above --max-missing-frac 0.2"):
        run_main(stubbed["tmp"])
    run_main(stubbed["tmp"], "--max-missing-frac", "0.9")  # the same shrinkage, accepted explicitly


def test_consecutive_missing_panos_abort(stubbed, monkeypatch):
    monkeypatch.setattr(dg, "fetch_panorama", lambda pano_id, info=None: None)
    monkeypatch.setattr(dg, "MAX_CONSECUTIVE_MISSING_PANOS", 3)
    with pytest.raises(SystemExit, match="3 panoramas in a row"):
        run_main(stubbed["tmp"])


def test_incomplete_pano_is_a_status(stubbed, monkeypatch):
    def incomplete(pano_id, info=None):
        raise dg.IncompletePanorama("missing tiles")
    monkeypatch.setattr(dg, "fetch_panorama", incomplete)
    with pytest.raises(SystemExit, match="no_pano/incomplete_pano"):  # all missing -> non-zero exit
        run_main(stubbed["tmp"])
    m = pd.read_csv(stubbed["tmp"] / "out" / "CurbRamp" / "manifest.csv")
    assert set(m.status) == {"incomplete_pano"}


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

def _maybe_missing(missing):
    return lambda pano_id, info=None: None if pano_id in missing else np.full((1024, 2048, 3), 90, np.uint8)


def test_resume_after_one_missing_pano_passes(stubbed, monkeypatch, capsys):
    """1 of 5 labels no_pano (20%, not above 0.2): the first run and a plain resume both exit 0. The resume retries
    only that label, which fails again, and is judged on the cumulative manifest (still 1 of 5)."""
    monkeypatch.setattr(dg, "fetch_panorama", _maybe_missing({"p3"}))
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
    so does the resume, which fetches exactly those four back to back."""
    rows = ([label(i, 0, 3, pano=f"p{i:02d}") for i in range(1, 11)]
            + [label(i, 5, 0, pano=f"p{i:02d}") for i in range(11, 21)])
    write_city(tmp_path, rows)
    monkeypatch.setattr(dg, "CITIES", {"toy-city": "toy"})
    monkeypatch.setattr(dg, "load_depth_model", lambda e, d: FakeDepth())
    monkeypatch.setattr(dg, "fetch_panorama", _maybe_missing({"p02", "p05", "p08", "p12"}))
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


def test_priors_are_not_written_on_a_refused_resume(stubbed):
    tmp = stubbed["tmp"]
    run_main(tmp)
    priors = tmp / "out" / "deployment_priors.json"
    assert priors.exists()
    priors.unlink()
    with pytest.raises(SystemExit, match="Refusing to resume"):
        run_main(tmp, "--tilt-beta", "0.5")
    assert not priors.exists()


def test_probe_retries_a_failed_diagonal_tile(monkeypatch):
    monkeypatch.setattr(dg, "_http_get", tile_server(fail_first={(7, 4): 1}))  # on the diagonal from (5, 2)
    assert REAL_FETCH("p").shape == (4096, 8192, 3)
