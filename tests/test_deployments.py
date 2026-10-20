"""The public-only naming rule shared by the analysis scripts (analysis/deployments.py)."""
import doctest
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("deployments", os.path.join(HERE, "..", "analysis", "deployments.py"))
deployments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deployments)


def test_doctests():
    assert doctest.testmod(deployments).failed == 0


def test_prefix_match_with_a_private_sibling_is_not_named():
    cities = {"x-y": "public", "x-z": "private"}
    assert deployments.city_label("x", cities) is None


def test_exact_match_wins_over_prefix_matches():
    assert deployments.city_label("x", {"x": "public", "x-z": "private"}) == "x"
    assert deployments.city_label("x", {"x": "private", "x-y": "public"}) is None


def test_unlisted_and_empty_are_never_named():
    assert deployments.city_label("x", {}) is None
    assert deployments.city_label("x", {"xy-z": "public"}) is None      # 'xy-z' is not a prefix match of 'x'


def test_underscores_and_all_public_prefixes():
    assert deployments.city_label("new_york", {"new-york-ny": "public"}) == "new_york"


def test_load_cities_accepts_api_response_and_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"status": "OK", "cities": [{"city_id": "a-b", "visibility": "public"}]}))
    assert deployments.load_cities(str(p)) == {"a-b": "public"}
    p.write_text(json.dumps([{"city_id": "a-b", "visibility": "private"}]))
    assert deployments.load_cities(str(p)) == {"a-b": "private"}
    assert deployments.load_cities(None) == {}
