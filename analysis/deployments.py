"""Which deployments may be named in anything committed: the public-only rule shared by the analysis scripts.

A city schema ``sidewalk_<x>`` is matched against the platform's city list (a saved ``/v3/api/cities`` response,
``curl https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > cities.json``). Schema names are short (``seattle``)
while ``city_id`` values often carry a region suffix (``seattle-wa``), so a prefix match is needed, and a prefix can
match more than one entry. The rule:

* an exact ``city_id == slug`` match decides on its own: named only if that entry is public;
* otherwise every entry whose ``city_id`` starts with ``slug + "-"`` is a candidate, and the schema is named only if
  there is at least one candidate and **all** candidates are public;
* a schema with no match at all (absent from the list) is never named.

So one public ``x-y`` next to a private ``x-z`` keeps schema ``x`` unnamed: when in doubt, do not name.

Example::

    >>> cities = {"x-y": "public", "x-z": "private", "seattle-wa": "public"}
    >>> city_label("x", cities) is None, city_label("seattle", cities)
    (True, 'seattle')
"""
import json


def load_cities(cities_json):
    """``{city_id: visibility}`` from a saved ``/v3/api/cities`` response (a dict with ``cities`` or a bare list);
    ``{}`` when no file is given, in which case nothing is ever named."""
    if not cities_json:
        return {}
    with open(cities_json) as f:
        cities = json.load(f)
    if isinstance(cities, dict):
        cities = cities.get("cities", [])
    return {c["city_id"]: c.get("visibility", "private") for c in cities}


def city_label(schema_city, cities):
    """The schema's city name when the rule in the module docstring allows naming it, else ``None``.

    ``schema_city`` is the schema name without ``sidewalk_``; ``cities`` is :func:`load_cities`'s output.

    >>> city_label("x", {"x": "private", "x-y": "public"}) is None
    True
    >>> city_label("x", {"x-y": "public", "x-w": "public"})
    'x'
    """
    slug = schema_city.replace("_", "-")
    if slug in cities:
        return schema_city if cities[slug] == "public" else None
    matches = [v for cid, v in cities.items() if cid.startswith(slug + "-")]
    if matches and all(v == "public" for v in matches):
        return schema_city
    return None
