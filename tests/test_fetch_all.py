"""analysis/fetch_all.py with a stubbed HTTP client (CI has no network): the default pull is unchanged and
--label-types writes a type-filtered cache in the same layout (issue #42)."""
import doctest
import json

import pytest
import requests

from analysis import fetch_all as fa

HEADER = "label_id,label_type,latitude,longitude\n"
CITIES = {"cities": [
    {"city_id": "alpha", "visibility": "public", "url": "https://sidewalk-alpha.cs.washington.edu/",
     "bounds": {"north": 1, "south": 0, "east": 1, "west": 0}},
    {"city_id": "beta", "visibility": "public", "url": "https://sidewalk-beta.cs.washington.edu",
     "bounds": {"north": 1, "south": 0, "east": 1, "west": 0}},
    {"city_id": "hidden", "visibility": "private", "url": None},
]}
EXPORTS = {  # city -> {label type -> csv body rows}
    "alpha": {"NoSidewalk": "1,NoSidewalk,0.1,0.1\n", "Signal": "2,Signal,0.2,0.2\n", "CurbRamp": "3,CurbRamp,0,0\n"},
    "beta": {"NoSidewalk": "7,NoSidewalk,0.5,0.5\n", "Signal": "", "CurbRamp": "8,CurbRamp,0,0\n"},
}


class Response:
    def __init__(self, text=None, payload=None):
        self.text, self._payload = text, payload

    def json(self):
        return self._payload


def stub(calls, truncate=()):
    """A requests.get stand-in serving CITIES and EXPORTS; whole-city exports of cities in ``truncate`` come back cut
    off mid-row, so the tiles are used."""
    def get(url, timeout=None):
        calls.append(url)
        if url == fa.CITIES_API:
            return Response(payload=CITIES)
        city = url.split("//sidewalk-")[1].split(".")[0]
        lt = url.split("labelType=")[1].split("&")[0] if "labelType=" in url else None
        body = "".join(EXPORTS[city].values()) if lt is None else EXPORTS[city][lt]
        if "bbox=" in url:
            # every tile returns the whole body: duplicates must be removed
            return Response(text=HEADER + body)
        if city in truncate:
            return Response(text=HEADER + body + '9,"unterminated')
        return Response(text=HEADER + body)
    return get


def test_default_pull_is_the_full_export_with_the_old_listing(tmp_path):
    calls = []
    fa.main([str(tmp_path)], get=stub(calls))
    assert [u for u in calls if "rawLabels" in u] == [
        "https://sidewalk-alpha.cs.washington.edu/v3/api/rawLabels?filetype=csv",
        "https://sidewalk-beta.cs.washington.edu/v3/api/rawLabels?filetype=csv"]
    assert (tmp_path / "alpha.csv").read_text() == HEADER + "".join(EXPORTS["alpha"].values())
    listing = json.loads((tmp_path / "cities.json").read_text())
    assert set(listing) == {"fetched_on", "url", "cities"}  # no label_types key: byte-identical to before
    assert [c["city_id"] for c in listing["cities"]] == ["alpha", "beta"]
    calls.clear()
    fa.main([str(tmp_path)], get=stub(calls))  # cached files are not fetched again
    assert not [u for u in calls if "rawLabels" in u]


def test_default_pull_tiles_a_truncated_export(tmp_path, capsys):
    calls = []
    fa.main([str(tmp_path)], get=stub(calls, truncate={"beta"}))
    assert len([u for u in calls if "beta" in u and "bbox=" in u]) == 16
    assert (tmp_path / "beta.csv").read_text() == HEADER + "7,NoSidewalk,0.5,0.5\n8,CurbRamp,0,0\n"  # deduplicated
    assert "beta: tiled, 2 unique labels" in capsys.readouterr().out


def test_label_types_write_one_csv_per_city_and_record_the_types(tmp_path, capsys):
    calls = []
    fa.main([str(tmp_path), "--label-types", "NoSidewalk,Signal"], get=stub(calls, truncate={"beta"}))
    raw = [u for u in calls if "rawLabels" in u and "bbox=" not in u]
    assert raw == [f"https://sidewalk-{c}.cs.washington.edu/v3/api/rawLabels?filetype=csv&labelType={t}"
                   for c in ("alpha", "beta") for t in ("NoSidewalk", "Signal")]
    assert all("labelType=" in u for u in calls if "bbox=" in u)  # the tiles are filtered too
    assert (tmp_path / "alpha.csv").read_text() == HEADER + "1,NoSidewalk,0.1,0.1\n2,Signal,0.2,0.2\n"
    assert (tmp_path / "beta.csv").read_text() == HEADER + "7,NoSidewalk,0.5,0.5\n"  # Signal: header only
    assert json.loads((tmp_path / "cities.json").read_text())["label_types"] == ["NoSidewalk", "Signal"]


def test_a_cache_holds_one_set_of_label_types(tmp_path):
    fa.main([str(tmp_path), "--label-types", "NoSidewalk"], get=stub([]))
    with pytest.raises(SystemExit, match="label types \\['NoSidewalk'\\]; this pull asks for all"):
        fa.main([str(tmp_path)], get=stub([]))
    with pytest.raises(SystemExit, match="asks for \\['Signal'\\]"):
        fa.main([str(tmp_path), "--label-types", "Signal"], get=stub([]))
    fa.main([str(tmp_path), "--label-types", "NoSidewalk"], get=stub([]))  # the same set: tops up


def test_data_generation_reads_a_filtered_cache(tmp_path):
    import data_generation as dg
    fa.main([str(tmp_path), "--label-types", "NoSidewalk"], get=stub([]))
    listing, source = dg.load_city_listing(str(tmp_path))
    assert list(dg.discover_cities(listing)) == ["alpha", "beta"] and source["from"] == "cache"
    assert dg.load_city("alpha", cache_dir=str(tmp_path))["label_type"].tolist() == ["NoSidewalk"]


def test_request_errors_fall_back_to_tiles(tmp_path, capsys):
    def get(url, timeout=None):
        if url == fa.CITIES_API:
            return Response(payload={"cities": CITIES["cities"][:1]})
        if "bbox=" not in url:
            raise requests.ConnectionError("dropped")
        return Response(text=HEADER + "1,NoSidewalk,0.1,0.1\n")
    fa.main([str(tmp_path), "--label-types", "NoSidewalk"], get=get)
    assert "whole-city request failed (dropped), tiling" in capsys.readouterr().out
    assert (tmp_path / "alpha.csv").read_text() == HEADER + "1,NoSidewalk,0.1,0.1\n"


def test_doctests():
    result = doctest.testmod(fa)
    assert result.attempted > 0 and result.failed == 0
