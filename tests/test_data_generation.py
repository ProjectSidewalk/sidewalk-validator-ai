"""data_generation.py on a CPU: label selection, AI-author handling, deployment priors, the resumable manifest."""
import doctest
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import crop_pose
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
    incorrect, 4 contested/unvalidated, 3 AI-authored correct, 2 AI-authored incorrect, 1 non-GSV with a bad pano
    size, 1 legacy negative pano_y, 1 other label type. AI votes on 5 rows would flip them to correct if counted."""
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
    rows[-3]["pano_width"] = 0     # the non-GSV row is dropped for its size, not its provider (#72)
    assert len(rows) == 30
    path = tmp_path / "toy-city.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def write_listing(cache, entries, **meta):
    """A /v3/api/cities listing as ``<cache>/cities.json``: ``entries`` are (city_id, visibility, url)."""
    listing = {**meta, "cities": [{"city_id": c, "visibility": v, "url": u} for c, v, u in entries]}
    (cache / dg.CITIES_CACHE_FILE).write_text(json.dumps(listing))
    return listing


# The shape of the live response (trimmed to the fields discovery reads, plus a few it ignores), deliberately unsorted:
# public with url, public with a non-UW url, private with and without url (invented ids), public without url (with and
# without a HOST_OVERRIDES entry, which the tests patch in).
CITIES_RESPONSE = {"status": "OK", "cities": [
    {"city_id": "teaneck-nj", "visibility": "public", "url": "https://sidewalk-teaneck.cs.washington.edu",
     "city_name_short": "Teaneck", "bounds": {"north": 1, "south": 0, "east": 1, "west": 0}},
    {"city_id": "private-a", "visibility": "private", "url": None},
    {"city_id": "seattle-wa", "visibility": "public", "url": "https://sidewalk-sea.cs.washington.edu/"},
    {"city_id": "private-b", "visibility": "private", "url": "https://sidewalk-private-b.cs.washington.edu"},
    {"city_id": "elsewhere", "visibility": "public", "url": "https://sidewalk.example.org/"},
    {"city_id": "overridden", "visibility": "public", "url": None},
    {"city_id": "nowhere", "visibility": "public", "url": None},
    {"city_id": "amsterdam", "visibility": "public", "url": "https://sidewalk-amsterdam.cs.washington.edu"},
]}
DEFAULT = ["amsterdam", "elsewhere", "seattle-wa", "teaneck-nj"]


@pytest.fixture(autouse=True)
def _no_listing_sleep(monkeypatch):
    monkeypatch.setattr(dg, "LISTING_RETRY_S", 0)


def counting_fetch(response=CITIES_RESPONSE):
    calls = []

    def fetch(url):
        calls.append(url)
        if isinstance(response, Exception):
            raise response
        return response
    return fetch, calls


def test_discover_cities_public_with_url_sorted(capsys, monkeypatch):
    cities = dg.discover_cities(CITIES_RESPONSE)
    assert cities == {"amsterdam": "amsterdam", "elsewhere": "https://sidewalk.example.org", "seattle-wa": "sea",
                      "teaneck-nj": "teaneck"}
    assert list(cities) == sorted(cities)
    assert "nowhere" in capsys.readouterr().out  # a public deployment without a url is named, not silently dropped
    shuffled = {"cities": CITIES_RESPONSE["cities"][::-1]}
    assert list(dg.discover_cities(shuffled).items()) == list(cities.items())  # independent of the API's order
    assert dg.discover_cities(CITIES_RESPONSE["cities"]) == cities  # an old fetch_all.py bare-list cities.json
    every = dg.discover_cities(CITIES_RESPONSE, public_only=False)  # host lookup for --cities
    assert every["private-b"] == "private-b" and "private-a" not in every and "nowhere" not in every
    assert dg.deployment_base_url(cities["elsewhere"]) == "https://sidewalk.example.org"
    monkeypatch.setattr(dg, "HOST_OVERRIDES", {"overridden": "ovr"})  # the mechanism, kept with an empty table
    assert dg.discover_cities(CITIES_RESPONSE)["overridden"] == "ovr"


def test_discover_cities_rejects_or_warns_on_odd_entries(capsys):
    e = lambda **kw: {"city_id": "a", "visibility": "public", **kw}  # noqa: E731
    assert dg.discover_cities({"cities": [e(url="  https://sidewalk-a.cs.washington.edu \n")]}) == {"a": "a"}
    assert dg.discover_cities({"cities": [e(url="sidewalk-a.cs.washington.edu"), {**e(), "city_id": "b",
                                                                                 "url": "ftp://x"}]}) == {}
    assert "not an http(s) url" in capsys.readouterr().out
    assert dg.discover_cities({"cities": [e(visibility="Public", url="https://x.org"),
                                          {"city_id": "b", "url": "https://y.org"}]}) == {}
    assert "unknown visibility to a ('Public'), b (None)" in capsys.readouterr().out
    for bad in ({}, [], {"cities": []}, {"cities": {"a": 1}}, {"cities": [{"url": "https://x.org"}]},
                {"cities": [{"city_id": None}]}, {"cities": ["a"]}, {"cities": [e(), e()]}, "nonsense",
                {"cities": [{"city_id": 5}]}, {"cities": [{"city_id": "  "}]}, {"cities": [{"city_id": "../etc"}]},
                {"cities": [{"city_id": "a b"}]}, {"cities": [{"city_id": "Seattle-WA"}]},
                {"cities": [{"city_id": "a;rm"}]}):
        with pytest.raises(dg.ListingError):
            dg.discover_cities(bad)


def test_city_listing_cache_round_trip(tmp_path):
    fetch, calls = counting_fetch()
    listing, source = dg.load_city_listing(str(tmp_path), fetch=fetch)
    today = dg.date.today().isoformat()
    assert source == {"from": "api", "url": dg.CITIES_API, "fetched_on": today} and len(calls) == 1
    cached = json.loads((tmp_path / "cities.json").read_text())
    assert cached == {"fetched_on": today, "url": dg.CITIES_API, "cities": CITIES_RESPONSE["cities"]}
    again, source2 = dg.load_city_listing(str(tmp_path), fetch=fetch)
    assert len(calls) == 1  # read from the cache, no fetch
    assert source2 == {"from": "cache", "url": dg.CITIES_API, "fetched_on": today}
    assert dg.discover_cities(again) == dg.discover_cities(listing) == dg.discover_cities(CITIES_RESPONSE)
    dg.load_city_listing(str(tmp_path), refresh=True, fetch=fetch)
    assert len(calls) == 2
    dg.load_city_listing(None, fetch=fetch)  # no cache directory: always the API, nothing written
    assert len(calls) == 3


def test_city_listing_without_fetch_date_records_none(tmp_path):
    (tmp_path / "cities.json").write_text(json.dumps(CITIES_RESPONSE["cities"]))  # an old fetch_all.py list
    _, source = dg.load_city_listing(str(tmp_path), fetch=lambda url: pytest.fail("fetched"))
    assert source == {"from": "cache", "url": None, "fetched_on": None}


def test_listing_is_never_written_into_a_pre_discovery_cache(tmp_path):
    (tmp_path / "toy-city.csv").write_text("label_id\n")  # a cache from before discovery: CSVs, no cities.json
    fetch, calls = counting_fetch()
    with pytest.raises(dg.ListingError, match="before city discovery.*--refresh-cities"):
        dg.load_city_listing(str(tmp_path), fetch=fetch)
    assert calls == [] and not (tmp_path / "cities.json").exists()
    with pytest.raises(SystemExit, match="before city discovery"):
        dg.resolve_cities(None, str(tmp_path), fetch=fetch)
    assert not (tmp_path / "cities.json").exists()
    dg.load_city_listing(str(tmp_path), refresh=True, fetch=fetch)  # deliberately
    assert (tmp_path / "cities.json").exists()


def test_a_broken_listing_is_not_cached(tmp_path):
    for response in ({}, {"status": "OK", "cities": []}, [{"city_id": "a"}], ValueError("not JSON")):
        fetch, calls = counting_fetch(response)
        with pytest.raises(SystemExit, match="City list"):
            dg.resolve_cities(None, str(tmp_path), fetch=fetch)
        assert not (tmp_path / "cities.json").exists()
    assert len(calls) == 2  # the exception was retried once
    (tmp_path / "cities.json").write_text("{not json")
    with pytest.raises(SystemExit, match="--refresh-cities"):
        dg.resolve_cities(None, str(tmp_path), fetch=counting_fetch()[0])


def test_parse_cities_behaves_as_before():
    known = {"seattle-wa": "sea", "chicago-il": "chicago"}
    assert dg.parse_cities(None, known) == known and dg.parse_cities("", known) == known
    assert dg.parse_cities("seattle-wa, chicago-il=chi,,other-city,new-city", known) == \
        {"seattle-wa": "sea", "chicago-il": "chi", "other-city": "other-city", "new-city": "new-city"}
    assert dg.parse_cities("vancouver-wa=vancouver") == {"vancouver-wa": "vancouver"}


def test_resolve_cities_reads_the_listing_only_when_a_host_is_needed(tmp_path, capsys):
    fetch, calls = counting_fetch()
    (tmp_path / "seattle-wa.csv").write_text("label_id\n")
    assert dg.resolve_cities("seattle-wa,x=y", str(tmp_path), fetch=fetch) == \
        ({"seattle-wa": "seattle-wa", "x": "y"}, {"from": "--cities"})  # cached CSV / explicit host: offline
    assert calls == []
    assert dg.resolve_cities("teaneck-nj,private-b", str(tmp_path), fetch=fetch)[0] == \
        {"teaneck-nj": "teaneck", "private-b": "private-b"}
    assert len(calls) == 1 and not (tmp_path / "cities.json").exists()  # the --cities path never writes the listing
    down, down_calls = counting_fetch(OSError("unreachable"))
    assert dg.resolve_cities("teaneck-nj", None, fetch=down)[0] == {"teaneck-nj": "teaneck-nj"}  # falls back
    assert len(down_calls) == 2 and "using the city_id as the subdomain for teaneck-nj" in capsys.readouterr().out


def test_resolve_cities_default_and_blank_flag(tmp_path):
    fetch, calls = counting_fetch()
    for arg in (None, "", " , "):  # a blank --cities is the default list, as on master
        cities, source = dg.resolve_cities(arg, str(tmp_path), fetch=fetch)
        assert list(cities) == DEFAULT
    assert len(calls) == 1 and source["from"] == "cache"
    write_listing(tmp_path, [("private-b", "private", "https://sidewalk-private-b.cs.washington.edu")])
    with pytest.raises(SystemExit, match="no public deployment"):
        dg.resolve_cities(None, str(tmp_path), fetch=fetch)


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
    extra = labels.iloc[:6].copy()
    extra["label_id"] = [101, 102, 103, 104, 105, 106]
    extra["validations"] = [expert("Disagree"), expert("Agree", "Agree"), expert("Disagree", "Agree"),
                            json.dumps([{"user_id": "ex", "validation": "Disagree", "validator_type": "AI"}]),
                            expert("Disagree", "Unsure"),  # Unsure is not a vote the other way
                            # the margin rule decides this one; the expert vote must not relabel it
                            json.dumps(json.loads(_votes(4, 0)) + json.loads(expert("Agree")))]
    labels = pd.concat([labels, extra], ignore_index=True)

    base, _ = dg.eligible_labels(labels, "CurbRamp")
    assert set(base.class_source) == {"margin"} and not base.label_id.isin([101, 102, 103, 104]).any()

    with_ex, _ = dg.eligible_labels(labels, "CurbRamp", experts=dg.parse_experts("someone=ex"))
    added = with_ex[with_ex.class_source == "expert"].set_index("label_id").label_class.to_dict()
    assert added == {101: "incorrect", 102: "correct", 105: "incorrect"}  # 103 contradicted; 104 not a Human vote
    assert with_ex.set_index("label_id").class_source[106] == "margin"
    # Margin-rule labels are unchanged, and the prior ignores the expert-added ones.
    pd.testing.assert_frame_equal(with_ex[with_ex.class_source == "margin"].reset_index(drop=True),
                                  base.reset_index(drop=True))
    assert (dg.deployment_prior_entry(with_ex, ["toy-city"], 35, measured_on="x")
            == dg.deployment_prior_entry(base, ["toy-city"], 35, measured_on="x"))
    sel = dg.select_training_labels(with_ex)
    assert (sel.label_class == "incorrect").sum() == 8 and {101, 105} <= set(sel.label_id)


def test_geometry_bounds_are_dropped_and_counted(tmp_path):
    eligible, dropped = dg.eligible_labels(fixture_labels(tmp_path), "CurbRamp")
    assert dict(dropped) == {"bad_pano_size": 1, "pano_y_out_of_range": 1}
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


@pytest.mark.parametrize("extra, real", [([], True), (["--limit", "3"], False), (["--cities", "toy-city"], False),
                                         (["--cities", ""], True)])  # a blank --cities is the default list
def test_priors_written_only_by_full_runs(tmp_path, monkeypatch, extra, real):
    fixture_labels(tmp_path)  # writes toy-city.csv
    write_listing(tmp_path, [("toy-city", "public", "https://sidewalk-toy.cs.washington.edu")])
    dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path),
             "--out", str(tmp_path / "out"), *extra])
    full, part = tmp_path / "out" / "deployment_priors.json", tmp_path / "out" / "deployment_priors.partial.json"
    assert full.exists() == real and part.exists() == (not real)
    entry = json.loads((full if real else part).read_text())["CurbRamp"]
    assert (entry["correct"], entry["incorrect"], entry["cities"], entry["n_rows"]) == (12, 6, ["toy-city"], 29)
    # the city list's origin: the cached listing, or a non-blank --cities
    assert entry["city_source"]["from"] == ("--cities" if "toy-city" in extra else "cache")


def test_priors_entry_records_the_api_fetch(tmp_path, monkeypatch):
    fixture_labels(tmp_path)
    monkeypatch.setattr(dg, "_get_json", lambda url: {"cities": [
        {"city_id": "toy-city", "visibility": "public", "url": "https://sidewalk-toy.cs.washington.edu"}]})
    dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path),
             "--out", str(tmp_path / "out"), "--refresh-cities"])  # the cache holds a CSV: writing needs the flag
    entry = json.loads((tmp_path / "out" / "deployment_priors.json").read_text())["CurbRamp"]
    assert entry["cities"] == ["toy-city"]
    assert entry["city_source"] == {"from": "api", "url": dg.CITIES_API, "fetched_on": dg.date.today().isoformat()}
    assert (tmp_path / "cities.json").exists()  # cached for the next run


def test_listed_city_without_a_cached_csv_is_fatal(tmp_path, monkeypatch):
    fixture_labels(tmp_path)
    write_listing(tmp_path, [("toy-city", "public", "https://sidewalk-toy.cs.washington.edu"),
                             ("new-city", "public", "https://sidewalk-new.cs.washington.edu")])
    monkeypatch.setattr(dg, "load_city", lambda *a, **k: pytest.fail("rawLabels loaded"))
    with pytest.raises(SystemExit, match="no <city_id>.csv for 1 listed cities: new-city.*fetch_all"):
        dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path),
                 "--out", str(tmp_path / "out")])


def test_select_only_against_a_run_with_another_city_list_writes_partial_priors(tmp_path, capsys):
    """A master-era output dir (19-city run record, real priors) must keep its priors when --select-only runs with
    the discovered list."""
    fixture_labels(tmp_path)
    write_listing(tmp_path, [("toy-city", "public", "https://sidewalk-toy.cs.washington.edu")])
    out = tmp_path / "out"
    (out / "CurbRamp").mkdir(parents=True)
    (out / "CurbRamp" / "generation_run.json").write_text(json.dumps({"cities": ["toy-city", "old-city"]}))
    real = out / "deployment_priors.json"
    real.write_text(json.dumps({"CurbRamp": {"correct": 1, "incorrect": 1, "cities": ["toy-city", "old-city"]}}))
    before = real.read_text()
    dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path), "--out", str(out)])
    assert real.read_text() == before
    entry = json.loads((out / "deployment_priors.partial.json").read_text())["CurbRamp"]
    assert entry["cities"] == ["toy-city"]
    assert "records a different city list (added: none; removed: old-city)" in capsys.readouterr().out
    (out / "CurbRamp" / "generation_run.json").write_text(json.dumps({"cities": ["toy-city"]}))  # same list: real
    dg.main(["--label_type", "CurbRamp", "--select-only", "--rawlabels-cache", str(tmp_path), "--out", str(out)])
    assert json.loads(real.read_text())["CurbRamp"]["cities"] == ["toy-city"]


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
    assert dg.check_resume(str(run), dg.selection_hash(a), st) == (None, {}, {})  # no previous run
    run.write_text(json.dumps({"selection_sha256": dg.selection_hash(a), "settings": st}))
    assert dg.check_resume(str(run), dg.selection_hash(a), st) == (None, {}, {})
    assert dg.check_resume(str(run), dg.selection_hash(b), st)[0] == dg.selection_hash(a)
    assert dg.check_resume(str(run), dg.selection_hash(a), {**st, "tilt_beta": 0.5, "renderer": "numpy"})[1] == \
        {"tilt_beta": (0.95, 0.5), "renderer": ("torch", "numpy")}


def test_check_resume_detects_a_changed_city_list(tmp_path):
    a = pd.DataFrame({"label_uid": ["x:1"], "label_class": ["correct"]})
    st = {"tilt_beta": 0.95}
    run = tmp_path / "generation_run.json"
    run.write_text(json.dumps({"selection_sha256": dg.selection_hash(a), "settings": st, "cities": ["x", "y"]}))
    h = dg.selection_hash(a)  # same selection: a city with no selected label does not move the hash
    assert dg.check_resume(str(run), h, st, ["y", "x"]) == (None, {}, {})  # order does not matter
    assert dg.check_resume(str(run), h, st, ["x", "z", "w"]) == (None, {}, {"added": ["w", "z"], "removed": ["y"]})
    assert dg.check_resume(str(run), h, st) == (None, {}, {})  # not compared when no list is given


def test_check_resume_skips_depth_settings_only_when_both_runs_are_fixed_size(tmp_path):
    a = pd.DataFrame({"label_uid": ["x:1"], "label_class": ["correct"]})
    h = dg.selection_hash(a)
    depth = {"tilt_beta": 0.95, "depth_encoder": "vitb", "depth_dataset": "vkitti"}
    fixed = {"tilt_beta": 0.95, "fixed_crop_half_px": 300}
    run = tmp_path / "generation_run.json"
    run.write_text(json.dumps({"selection_sha256": h, "settings": {**depth, "fixed_crop_half_px": 300}}))
    assert dg.check_resume(str(run), h, fixed) == (None, {}, {})  # a fixed record that still names the encoder
    assert dg.check_resume(str(run), h, {**fixed, "fixed_crop_half_px": 200})[1] == {"fixed_crop_half_px": (300, 200)}
    run.write_text(json.dumps({"selection_sha256": h, "settings": depth}))
    assert dg.check_resume(str(run), h, fixed)[1] == {
        "depth_encoder": ("vitb", None), "depth_dataset": ("vkitti", None), "fixed_crop_half_px": (None, 300)}
    assert dg.check_resume(str(run), h, {**depth, "depth_encoder": "vitl"})[1] == {"depth_encoder": ("vitb", "vitl")}


def test_manifest_row_uid_and_crop_path():
    row = pd.Series({"city": "seattle-wa", "label_id": 12345, "pano_id": "abc", "label_class": "correct",
                     "author": "human", "human_agree": 3, "human_disagree": 0, "camera_pitch": np.nan})
    rec = dg.manifest_row(row, "CurbRamp", "correct/seattle-wa_12345.webp", 0.95)
    assert rec["label_uid"] == "seattle-wa:12345" and rec["tilt_beta"] == 0.95 and "hard" not in rec
    assert set(rec) == set(dg.MANIFEST_COLUMNS)


def test_manifest_row_records_the_pose_and_the_provider_beta():
    row = pd.Series({"city": "bayonne-fr", "label_id": 7, "pano_id": "x", "label_class": "correct", "author": "human",
                     "human_agree": 3, "human_disagree": 0, "pano_source": "panoramax", "camera_pitch": -4.0,
                     "camera_roll": 3.0})
    pose = crop_pose.pose_for_row(row, 0.95)
    rec = dg.manifest_row(row, "CurbRamp", "correct/bayonne-fr_7.webp", 0.95, pose)
    assert (rec["pose_source"], rec["pose_pitch"], rec["pose_roll"], rec["tilt_beta"], rec["click_lat_offset_deg"]) == \
        ("panoramax_db", 4.0, 3.0, 1.0, 0.0)
    mly = crop_pose.pose_for_row({**row, "pano_source": "mapillary"}, 0.95)
    assert dg.manifest_row({**row, "pano_source": "mapillary"}, "CurbRamp", "x.webp", 0.95, mly)["tilt_beta"] == 0.0
    assert rec["camera_pitch"] == -4.0  # the DB column as stored, in the provider's sign
    assert set(rec) == set(dg.MANIFEST_COLUMNS)


def test_valid_geometry_keeps_supported_providers_only():
    df = pd.DataFrame({"pano_source": ["gsv", "Mapillary", "panoramax", "infra3d", None], "pano_x": [1] * 5,
                       "pano_y": [5] * 5, "pano_width": [10] * 5, "pano_height": [10] * 5})
    keep, why = dg.valid_geometry(df)
    assert keep.tolist() == [True, True, True, False, False] and dict(why) == {"unsupported_source": 2}


ROTATION = [1.3857263583832, 0.71804330335161, -0.58250746512038]  # Richmond VA 2163793620710887: (2.2318, -6.355)


def _pose_row(src, pitch=None, roll=None, author="human", pano="p1"):
    return {"pano_source": src, "pano_id": pano, "camera_pitch": pitch, "camera_roll": roll, "author": author}


@pytest.mark.parametrize("row, rotations, want", [
    # GSV: exactly as before #72 (the DB columns unchanged, --tilt-beta); no column at all is no pose.
    (_pose_row("gsv", 2.0), None, (2.0, None, "gsv_db", 0.95, "db", 0.0)),
    (_pose_row("GSV", 2.0, float("nan")), None, (2.0, None, "gsv_db", 0.95, "db", 0.0)),
    (_pose_row("gsv"), None, (None, None, "none", None, "gsv_no_db_pose", 0.0)),
    # Mapillary: the click is rig-frame (beta 0); the pose is still read, pitch negated: DB first, else the API.
    (_pose_row("mapillary", 2.0, -6.0), None, (-2.0, -6.0, "mapillary_db", 0.0, "mapillary_click_rig_frame", 0.0)),
    (_pose_row("mapillary", 2.0, -6.0), {"p1": [0.0, 0.0, 0.0]},
     (-2.0, -6.0, "mapillary_db", 0.0, "mapillary_click_rig_frame", 0.0)),                      # DB wins
    (_pose_row("mapillary", 40.0, 40.0), None, (None, None, "none", None, "mapillary_db_over_tilt", 0.0)),
    (_pose_row("mapillary", 2.0, None), None, (None, None, "none", None, "mapillary_no_db_pose_api_off", 0.0)),
    (_pose_row("mapillary", None, -6.0), {"p1": ROTATION},                                     # one angle: not a pose
     (pytest.approx(-2.2318, abs=1e-4), pytest.approx(-6.355, abs=1e-4), "mapillary_api", 0.0,
      "mapillary_click_rig_frame", 0.0)),
    (_pose_row("mapillary"), None, (None, None, "none", None, "mapillary_no_db_pose_api_off", 0.0)),
    (_pose_row("mapillary"), {}, (None, None, "none", None, "mapillary_api_not_fetched", 0.0)),
    (_pose_row("mapillary"), {"p1": None}, (None, None, "none", None, "mapillary_api_missing", 0.0)),
    (_pose_row("mapillary"), {"p1": [math.pi, 0.0, 0.0]}, (None, None, "none", None, "mapillary_api_over_tilt", 0.0)),
    (_pose_row("mapillary"), {"p1": [0.0, 0.0, 0.0]}, (None, None, "none", None, "mapillary_api_zero_rotation", 0.0)),
    (_pose_row("mapillary"), {"p1": [1, "x", 2]}, (None, None, "none", None, "mapillary_api_malformed", 0.0)),
    # Panoramax: both angles non-zero is the only case the viewer levels (beta 1.0); pitch-only shifts the click.
    (_pose_row("panoramax", -4.0, 3.0), None, (4.0, 3.0, "panoramax_db", 1.0, "level_frame_click", 0.0)),
    (_pose_row("panoramax", 356.0, 3.0), None, (4.0, 3.0, "panoramax_db", 1.0, "level_frame_click", 0.0)),  # wrapped
    (_pose_row("panoramax", 0.0, 0.0), None, (None, None, "none", None, "panoramax_zero_or_absent_pose", 0.0)),
    (_pose_row("panoramax", 0.0, None), None, (None, None, "none", None, "panoramax_zero_or_absent_pose", 0.0)),
    (_pose_row("panoramax"), None, (None, None, "none", None, "panoramax_zero_or_absent_pose", 0.0)),
    (_pose_row("panoramax", 5.0, None), None,
     (-5.0, None, "panoramax_db", 0.0, "panoramax_pitch_only_click_offset", 5.0)),
    (_pose_row("panoramax", 5.0, 0.0), None,
     (-5.0, None, "panoramax_db", 0.0, "panoramax_pitch_only_click_offset", 5.0)),
    (_pose_row("panoramax", 0.0, 2.0), None,
     (None, 2.0, "panoramax_db", 0.0, "panoramax_roll_only_click_rig_frame", 0.0)),
    # AI labels were placed on the rig image: never corrected, whatever the provider.
    (_pose_row("mapillary", 2.0, 1.0, author="ai"), None, (None, None, "rig_frame", None, "ai_label", 0.0)),
    (_pose_row("gsv", 2.0, author="ai"), None, (None, None, "rig_frame", None, "ai_label", 0.0)),
    (_pose_row("infra3d", 1.0, 1.0), None, (None, None, "none", None, "unsupported_source:infra3d", 0.0)),
])
def test_pose_for_row(row, rotations, want):
    assert tuple(crop_pose.pose_for_row(row, 0.95, rotations)) == want


def test_check_resume_rejects_changed_source_beta_and_mapillary_api(tmp_path):
    """Both change non-GSV rows, so a resume across either is refused (review S4)."""
    run = tmp_path / "generation_run.json"
    st = {"tilt_beta": 0.95, "source_beta": {"mapillary": 0.0, "panoramax": 1.0}, "mapillary_api": False}
    run.write_text(json.dumps({"selection_sha256": "h", "settings": st}))
    assert dg.check_resume(str(run), "h", st)[1] == {}
    assert dg.check_resume(str(run), "h", {**st, "source_beta": {"mapillary": 0.0, "panoramax": 0.9}})[1] == \
        {"source_beta": ({"mapillary": 0.0, "panoramax": 1.0}, {"mapillary": 0.0, "panoramax": 0.9})}
    assert dg.check_resume(str(run), "h", {**st, "mapillary_api": True})[1] == {"mapillary_api": (False, True)}
    run.write_text(json.dumps({"selection_sha256": "h", "settings": {"tilt_beta": 0.95}}))  # a record from before #72
    assert dg.check_resume(str(run), "h", st)[1] == {}


def test_load_city_keeps_numeric_pano_ids_as_strings(tmp_path):
    pd.DataFrame({"label_id": [1, 2], "pano_id": ["1792391791667719", "0012"]}).to_csv(tmp_path / "c.csv", index=False)
    assert dg.load_city("c", cache_dir=str(tmp_path))["pano_id"].tolist() == ["1792391791667719", "0012"]


def test_mapillary_api_pose_is_the_port_negated():
    pose = crop_pose.pose_for_row(_pose_row("mapillary"), 0.95, {"p1": ROTATION})
    assert pose.pose_source == "mapillary_api" and pose.beta == 0.0
    assert (round(pose.camera_pitch, 4), round(pose.camera_roll, 4)) == (-2.2318, -6.355)
    assert crop_pose.needs_mapillary_rotation(_pose_row("mapillary"))
    assert not crop_pose.needs_mapillary_rotation(_pose_row("mapillary", 1.0, 2.0))
    assert not crop_pose.needs_mapillary_rotation(_pose_row("mapillary", author="ai"))
    assert not crop_pose.needs_mapillary_rotation(_pose_row("gsv"))


def test_every_pose_source_is_declared():
    assert set(crop_pose.POSE_SOURCES) >= {"gsv_db", "mapillary_db", "mapillary_api", "panoramax_db", "none",
                                           "rig_frame", "image_estimate"}


def test_fetch_mapillary_rotations_caches_answers_not_failures(tmp_path):
    class Resp:
        def __init__(self, status, body=None, text=""):
            self.status_code, self._body, self.text = status, body, text

        def json(self):
            if isinstance(self._body, Exception):
                raise self._body
            return self._body
    answers = {"a": Resp(200, {"id": "a", "computed_rotation": ROTATION}), "b": Resp(404, None, "not found"),
               "c": Resp(500), "d": Resp(404, None, '{"error":{"type":"OAuthException","code":190}}'),
               "e": Resp(200, {"error": {"code": 1}}), "f": Resp(200, ValueError("not JSON"))}
    asked = []

    class Session:
        def get(self, url, params=None, headers=None, timeout=None):
            pid = url.rsplit("/", 1)[1]
            asked.append(pid)
            return answers[pid]
    cache = str(tmp_path / "rot.json")
    rot, n, failed = crop_pose.fetch_mapillary_rotations(list("abcdefa"), cache, "tok", Session(), pause_s=0)
    assert rot == {"a": ROTATION, "b": None} and n == 6
    assert failed == [("c", "http_500"), ("d", "http_404"), ("e", "error_body"), ("f", "non_json_body")]
    assert "tok" not in open(cache).read()
    rot, n, failed = crop_pose.fetch_mapillary_rotations(["a", "b"], cache, "tok", Session(), pause_s=0)
    assert n == 0 and rot == {"a": ROTATION, "b": None} and asked.count("a") == 1


def test_pose_counts_by_provider_and_reason():
    rows = [_pose_row("panoramax", 0.0, 0.0), _pose_row("panoramax", 0.0, None), _pose_row("panoramax", 1.0, 1.0),
            _pose_row("panoramax", 1.0, None), _pose_row("gsv", 1.0)]
    poses = [crop_pose.pose_for_row(r, 0.95) for r in rows]
    assert dg.pose_counts(poses, [r["pano_source"] for r in rows]) == {
        "gsv": {"gsv_db/db": 1},
        "panoramax": {"none/panoramax_zero_or_absent_pose": 2, "panoramax_db/level_frame_click": 1,
                      "panoramax_db/panoramax_pitch_only_click_offset": 1}}


def test_depth_sized_crop_clips_near_features():
    img = np.zeros((2048, 2048, 3), np.uint8)
    crop, half, clipped = dg.depth_sized_crop(img, 2.0)  # 3050 px half-width > 1024
    assert clipped and half == 3050 and crop.shape == (640, 640, 3)
    with pytest.raises(ValueError):
        dg.depth_sized_crop(img, 0.0)


@pytest.mark.parametrize("module", [dg, hard_examples, crop_pose])
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


def test_fine_tune_records_a_fixed_crop_size_only_when_there_is_one(tmp_path):
    import fine_tune_classifier as ft
    base = tmp_path / "Toy"
    base.mkdir()
    (base / "generation_run.json").write_text(json.dumps({"tilt_beta": 0.95, "depth_encoder": "vitb"}))
    assert ft.generation_settings("Toy", str(tmp_path)) == {"tilt_beta": 0.95, "depth_encoder": "vitb"}
    (base / "generation_run.json").write_text(json.dumps({"tilt_beta": 0.95, "fixed_crop_half_px": 300}))
    assert ft.generation_settings("Toy", str(tmp_path)) == {"tilt_beta": 0.95, "depth_encoder": None,
                                                            "fixed_crop_half_px": 300}


def _write_priors_many(path, label_type):  # module level, so a spawned process can import it
    for i in range(40):
        dg.update_deployment_priors(path, label_type, {"correct": i, "incorrect": i})
