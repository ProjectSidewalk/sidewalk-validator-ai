"""regions.py: the country table, the US-state rule, naming of private-only countries, and what the module commits."""
import doctest
import json
import os

import pandas as pd
import pytest

import regions

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "public_cities_2026-11-05.txt")

LISTING = [
    {"city_id": "seattle-wa", "country_id": "usa", "visibility": "public", "city_name_short": "Seattle"},
    {"city_id": "amsterdam", "country_id": "netherlands", "visibility": "public"},
    {"city_id": "hidden-x", "country_id": "atlantis", "visibility": "private", "city_name_short": "Hiddenville",
     "city_name_formatted": "Hiddenville, Atlantis"},
    {"city_id": "quiet-y", "country_id": "netherlands", "visibility": "private"},
]


def test_table_prefers_listing_then_fallback_then_override():
    listing = [{"city_id": "cdmx", "country_id": "Elsewhere"}]   # the listing wins over the fallback table
    t = regions.country_table(listing, overrides={})
    assert t["cdmx"] == "elsewhere"
    assert t["amsterdam"] == "netherlands"                        # not in the listing: the fallback
    t = regions.country_table(listing, overrides={"cdmx": "mexico"})
    assert t["cdmx"] == "mexico"                                  # overrides win


def test_private_ids_resolve_from_listing_without_being_in_the_module():
    t = regions.country_table(LISTING, overrides={})
    assert "hidden-x" not in regions.COUNTRY_BY_PUBLIC_CITY
    assert regions.country_of("hidden-x", t) == "atlantis"


def test_override_file_from_env(tmp_path, monkeypatch):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"secret-city": "Lemuria", "_regions": {"lemuria": "Oceania"}}))
    monkeypatch.setenv(regions.OVERRIDES_ENV, str(path))
    assert regions.country_table([])["secret-city"] == "lemuria"
    assert "_regions" not in regions.country_table([])
    assert regions.region_of("lemuria") == "oceania"
    out = regions.assign(pd.DataFrame({"city": ["secret-city"]}), regions.country_table([]))
    assert out["region"].tolist() == ["oceania"]
    monkeypatch.delenv(regions.OVERRIDES_ENV)
    assert "secret-city" not in regions.country_table([])
    assert regions.region_of("lemuria") == regions.UNKNOWN


def test_us_state_suffix_heuristic_and_unknown():
    assert regions.country_of("nowhere-zz", {}) == "usa"
    assert regions.country_of("nowhere", {}) == regions.UNKNOWN
    assert regions.country_of("x-123", {}) == regions.UNKNOWN


def test_assign_raises_on_unknown_unless_allowed():
    df = pd.DataFrame({"city": ["amsterdam", "nowhere", "seattle-wa"]})
    t = regions.country_table([], {})
    with pytest.raises(ValueError, match="nowhere"):
        regions.assign(df, t)
    out = regions.assign(df, t, allow_unknown=True)
    assert out["country"].tolist() == ["netherlands", "unknown", "usa"]
    assert out["region"].tolist() == ["europe", "unknown", "north-america"]


def test_display_names_letter_private_only_countries():
    public = regions.public_countries(LISTING)
    assert public == {"usa", "netherlands"}
    names = regions.display_names(["usa", "netherlands", "atlantis", "lemuria"], public)
    assert names == {"atlantis": "private country A", "lemuria": "private country B", "netherlands": "netherlands",
                     "usa": "usa"}


def test_non_public_terms_cover_private_names_and_private_only_countries():
    terms = regions.non_public_terms(LISTING)
    assert {"hidden-x", "Hiddenville", "Hiddenville, Atlantis", "atlantis", "quiet-y"} <= terms
    assert "netherlands" not in terms   # it has a public deployment
    assert "seattle-wa" not in terms


def test_region_of_every_table_country_is_known():
    for country in set(regions.COUNTRY_BY_PUBLIC_CITY.values()):
        assert regions.region_of(country, regions.REGION_BY_COUNTRY) != regions.UNKNOWN


def test_region_table_names_only_countries_with_a_public_deployment():
    assert set(regions.REGION_BY_COUNTRY) == set(regions.COUNTRY_BY_PUBLIC_CITY.values())


def test_listing_without_country_id_is_refused(tmp_path):
    path = tmp_path / "cities.json"
    path.write_text(json.dumps({"cities": [{"city_id": "a", "visibility": "public"}]}))
    with pytest.raises(ValueError, match="no country_id"):
        regions.load_listing(str(path))
    path.write_text(json.dumps(LISTING))   # a bare list is accepted
    assert len(regions.load_listing(str(path))) == 4


def test_load_listing_live_path_uses_the_stubbed_fetch():
    entries = regions.load_listing(None, fetch=lambda url: {"cities": LISTING})
    assert [c["city_id"] for c in entries] == [c["city_id"] for c in LISTING]


def test_module_commits_no_private_name():
    with open(FIXTURE) as f:
        public = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    assert set(regions.COUNTRY_BY_PUBLIC_CITY) <= public


def test_doctests():
    assert doctest.testmod(regions).failed == 0
