"""Country and region of every Project Sidewalk deployment, and how to name them in public text.

The cross-region study (``analysis/region_transfer.py``, issues #33, #52 and #15) holds out one country at a time and
trains a validator that is told the country or region of each label (``region_model.py``). Both need a country per
``city_id``. The source of truth is the ``country_id`` field of ``/v3/api/cities``, which every deployment carries,
private ones included. The table below is a fallback for public deployments only, so a saved listing that lacks an
entry still resolves; private deployments come from the listing or from a private override file (env
``REGION_OVERRIDES``, a JSON ``{"<city_id>": "<country_id>", "_regions": {"<country_id>": "<region>"}}`` kept outside
the repository), never from this module. ``REGION_BY_COUNTRY`` likewise lists only countries with a public deployment:
the region of a country whose deployments are all private comes from the override file's ``_regions`` (without it,
that country's region is ``UNKNOWN``, which ``region_model`` never makes a vocabulary entry: at the region level
those rows get the ``<unk>`` row, in training and at inference, the same row an unseen country is scored with).

Naming rule for public text (the one ``publish_to_hf.py`` enforces for model cards): a country is named only if at
least one of its deployments is public. A country whose deployments are all private is "private country A", "B", and
so on, in the sorted order of its id, because naming it would point at the private deployment.

Example::

    >>> listing = [{"city_id": "seattle-wa", "country_id": "usa", "visibility": "public"},
    ...            {"city_id": "hidden-x", "country_id": "atlantis", "visibility": "private"}]
    >>> table = country_table(listing, overrides={})
    >>> country_of("hidden-x", table), region_of(country_of("seattle-wa", table), REGION_BY_COUNTRY)
    ('atlantis', 'north-america')
    >>> display_names(["usa", "atlantis"], public_countries(listing))
    {'atlantis': 'private country A', 'usa': 'usa'}
"""
import json
import os

UNKNOWN = "unknown"
LEVELS = ("country", "region")
OVERRIDES_ENV = "REGION_OVERRIDES"

# The 46 public deployments of the live /v3/api/cities on 2026-11-05. A fallback only: the listing wins.
COUNTRY_BY_PUBLIC_CITY = {
    "amsterdam": "netherlands", "auckland": "new-zealand", "bayonne-fr": "france", "burnaby": "canada",
    "cdmx": "mexico", "chandigarh-india": "india", "cuenca": "ecuador", "la-piedad": "mexico",
    "rancagua-chile": "chile", "santiago-chile": "chile", "sao-paulo-brazil": "brazil", "spgg": "mexico",
    "blackhawk-hills-il": "usa", "chicago-il": "usa", "cliffside-park-nj": "usa", "clifton-nj": "usa",
    "columbus-oh": "usa", "danville-il": "usa", "detroit-mi": "usa", "fort-wayne-in": "usa",
    "gainesville-fl": "usa", "hackensack-nj": "usa", "houston-tx": "usa", "knox-oh": "usa", "la-ca": "usa",
    "laurens-ia": "usa", "madison-wi": "usa", "maywood-nj": "usa", "mendota-il": "usa", "newberg-or": "usa",
    "newport-ky": "usa", "niagara-falls-ny": "usa", "oradell-nj": "usa", "paterson-nj": "usa",
    "pittsburgh-pa": "usa", "richmond-va": "usa", "seattle-wa": "usa", "st-louis-mo": "usa", "teaneck-nj": "usa",
    "tucson-az": "usa", "vancouver-wa": "usa", "virden-il": "usa", "walla-walla-wa": "usa", "waltham-ma": "usa",
    "washington-dc": "usa", "west-chester-pa": "usa",
}

# Countries with at least one public deployment on 2026-11-05 only (see the module docstring for the others).
REGION_BY_COUNTRY = {
    "usa": "north-america", "canada": "north-america",
    "mexico": "latin-america", "ecuador": "latin-america", "chile": "latin-america", "brazil": "latin-america",
    "netherlands": "europe", "france": "europe",
    "india": "asia",
    "new-zealand": "oceania",
}
REGIONS_KEY = "_regions"


def _entries(listing):
    """The list of city entries of a listing: ``{"cities": [...]}`` or a bare list."""
    return listing.get("cities", []) if isinstance(listing, dict) else list(listing)


def load_listing(path=None, fetch=None):
    """The ``/v3/api/cities`` entries (a list of dicts) from a saved file or, with ``path=None``, the live API.

    A saved file may be ``{"cities": [...]}`` (as ``data_generation.load_city_listing`` writes it) or a bare list.
    Raises ``ValueError`` when an entry has no ``country_id`` (an old saved listing). ``fetch`` is passed to
    ``data_generation.fetch_city_listing`` (tests stub the network with it).
    """
    if path is None:
        from data_generation import fetch_city_listing
        listing = fetch_city_listing(fetch=fetch)
    else:
        with open(path) as f:
            listing = json.load(f)
    entries = _entries(listing)
    missing = [str(c.get("city_id")) for c in entries if not c.get("country_id")]
    if missing:
        raise ValueError(f"listing has no country_id for {len(missing)} deployment(s); re-fetch: "
                         "curl -s https://sidewalk-teaneck.cs.washington.edu/v3/api/cities > cities.json")
    return entries


def _overrides_from_env():
    path = os.environ.get(OVERRIDES_ENV)
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def country_table(listing, overrides=None):
    """``{city_id: country_id}``: the listing first (public and private), then ``COUNTRY_BY_PUBLIC_CITY`` for ids
    the listing lacks, then ``overrides`` (a dict; ``None`` reads the JSON file named by env ``REGION_OVERRIDES``),
    which win. Values are lower-cased.

    >>> country_table([{"city_id": "a", "country_id": "X"}], overrides={"b": "y"})["a"]
    'x'
    """
    table = {k: v.lower() for k, v in COUNTRY_BY_PUBLIC_CITY.items()}
    for c in _entries(listing):
        if c.get("country_id"):
            table[str(c["city_id"])] = str(c["country_id"]).lower()
    extra = _overrides_from_env() if overrides is None else overrides
    table.update({str(k): str(v).lower() for k, v in extra.items() if k != REGIONS_KEY})
    return table


def country_of(city_id, table):
    """The country of ``city_id``: the table, else ``usa`` when the id ends in a two-letter state code (the rule
    the repository already uses, e.g. ``seattle-wa``), else ``UNKNOWN``.

    The suffix rule is a heuristic, safe only because the table normally holds every deployment: a non-US id that
    ends in two letters and is missing from the table would come back as ``usa`` (``bayonne-fr`` without its row in
    ``COUNTRY_BY_PUBLIC_CITY`` would). That is why callers build the table from a saved ``/v3/api/cities`` listing,
    which carries ``country_id`` for every deployment, private ones included.

    >>> country_of("nowhere-zz", {}), country_of("nowhere", {}), country_of("x-123", {})
    ('usa', 'unknown', 'unknown')
    """
    city_id = str(city_id)
    if city_id in table:
        return table[city_id]
    last = city_id.split("-")[-1]
    if "-" in city_id and len(last) == 2 and last.isalpha():
        return "usa"
    return UNKNOWN


def region_table(overrides=None):
    """``REGION_BY_COUNTRY`` plus the ``_regions`` of ``overrides`` (``None`` reads env ``REGION_OVERRIDES``)."""
    extra = _overrides_from_env() if overrides is None else overrides
    return {**REGION_BY_COUNTRY, **{str(k).lower(): str(v).lower() for k, v in extra.get(REGIONS_KEY, {}).items()}}


def region_of(country, table=None):
    """The world region of a country id (``table``, default ``region_table()``), or ``UNKNOWN``.

    >>> region_of("mexico", REGION_BY_COUNTRY), region_of("atlantis", REGION_BY_COUNTRY)
    ('latin-america', 'unknown')
    >>> region_of("atlantis", region_table({"_regions": {"atlantis": "oceania"}}))
    'oceania'
    """
    return (region_table() if table is None else table).get(country, UNKNOWN)


def assign(df, table, allow_unknown=False):
    """A copy of ``df`` with ``country`` and ``region`` columns from its ``city`` column. Raises ``ValueError``
    naming the ids without a country unless ``allow_unknown`` (they then stay ``UNKNOWN``).

    >>> import pandas as pd
    >>> assign(pd.DataFrame({"city": ["cdmx"]}), country_table([], {}))[["country", "region"]].values.tolist()
    [['mexico', 'latin-america']]
    """
    out = df.copy()
    cities = out["city"].astype(str)
    lookup = {c: country_of(c, table) for c in cities.unique()}
    unknown = sorted(c for c, v in lookup.items() if v == UNKNOWN)
    if unknown and not allow_unknown:
        raise ValueError(f"no country for {len(unknown)} deployment(s): {unknown[:10]}; add them to the listing or "
                         f"to the {OVERRIDES_ENV} file")
    out["country"] = cities.map(lookup)
    rtable = region_table()
    out["region"] = out["country"].map(lambda c: region_of(c, rtable))
    return out


def public_countries(listing):
    """Countries with at least one deployment the listing marks public."""
    return {str(c["country_id"]).lower() for c in _entries(listing)
            if c.get("visibility") == "public" and c.get("country_id")}


def display_names(countries, public):
    """How each country is named in public text: by id when in ``public``, else "private country A", "B", ... in
    the sorted order of the ids.

    >>> display_names(["b", "a", "usa"], {"usa"})
    {'a': 'private country A', 'b': 'private country B', 'usa': 'usa'}
    """
    names = {}
    private = sorted(c for c in set(countries) if c not in public)
    for i, c in enumerate(private):
        names[c] = f"private country {chr(ord('A') + i)}"
    for c in set(countries):
        if c in public:
            names[c] = c
    return dict(sorted(names.items()))


def non_public_terms(listing):
    """Strings that must never appear in committed text: every non-public deployment's ``city_id``,
    ``city_name_short`` and ``city_name_formatted``, and every country whose deployments are all non-public (the
    term set ``publish_to_hf.py`` checks model cards against)."""
    entries = _entries(listing)
    public = public_countries(entries)
    terms = set()
    for c in entries:
        if c.get("visibility") == "public":
            continue
        for key in ("city_id", "city_name_short", "city_name_formatted"):
            if c.get(key):
                terms.add(str(c[key]))
        if c.get("country_id") and str(c["country_id"]).lower() not in public:
            terms.add(str(c["country_id"]).lower())
    return terms
