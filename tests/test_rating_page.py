"""Tests for analysis/rating_page.py: the generated page's structure and embedded data (the JS is checked by hand)."""
import doctest
import json
import re
from html.parser import HTMLParser

import pytest

from analysis import rating_page as rp


def _page(**kw):
    args = dict(title="T", heading="H", intro_html="<p>i</p>",
                items=[{"id": "p00000a", "image": "items/001_p00000a.jpg", "alt": "Item 1", "caption_html": "1"},
                       {"id": "p00000b", "image": "items/002_p00000b.jpg", "alt": "Item 2", "caption_html": "2",
                        "extra_html": "<p>extra &amp; more</p>"}],
                fields=[{"name": "v", "legend": "V?", "kind": "radio", "options": [("yes", "Yes", "y"),
                                                                                      ("no", "No", "n")]},
                        {"name": "notes", "legend": "Notes", "kind": "text"}],
                sheet_columns=["packet_id", "label_type", "v", "notes"],
                fixed={"p00000a": {"label_type": "CurbRamp"}, "p00000b": {"label_type": "Obstacle"}},
                storage_key="k", export_name="sheet")
    args.update(kw)
    return rp.render_page(**args)


def meta(page):
    return json.loads(re.search(r'<script id="meta" type="application/json">(.*?)</script>', page, re.S)
                      .group(1))


def test_doctests():
    res = doctest.testmod(rp)
    assert res.failed == 0 and res.attempted > 0


def test_meta_rows_keep_sheet_order_and_fixed_columns_only():
    m = meta(_page())
    assert m["columns"] == ["packet_id", "label_type", "v", "notes"]
    assert m["rows"] == [{"packet_id": "p00000a", "label_type": "CurbRamp"},
                         {"packet_id": "p00000b", "label_type": "Obstacle"}]
    assert m["gate"] is None


def test_page_parses_and_has_one_card_and_one_radio_group_per_item():
    page = _page()

    class P(HTMLParser):
        cards, radios, open_tags = 0, [], []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "section" and a.get("class") == "card":
                P.cards += 1
            if tag == "input" and a.get("type") == "radio":
                P.radios.append(a["name"])

    P().feed(page)
    assert P.cards == 2
    assert sorted(set(P.radios)) == ["p00000a__v", "p00000b__v"] and len(P.radios) == 4
    assert "<kbd>Y</kbd>" in page and 'id="cards">' in page


def test_gate_hides_cards_and_is_in_meta():
    page = _page(gate={"column": "my_verdict", "values": ["correct"], "html": "<p>load it</p>"})
    assert '<div id="cards" hidden>' in page and 'id="gatefile"' in page
    assert meta(page)["gate"] == {"column": "my_verdict", "values": ["correct"]}


@pytest.mark.parametrize("text", ["</script><b>x", "<!--<script>", "<!-- a", "a & b > c", "<\\/script>"])
def test_data_cannot_close_or_reopen_a_script_element(text):
    page = _page(fixed={"p00000a": {"label_type": text}, "p00000b": {}})
    data = re.search(r'<script id="meta" type="application/json">(.*?)</script>', page, re.S).group(1)
    assert "<" not in data and ">" not in data and "&" not in data
    assert page.count("<script") == 2 and page.count("</script>") == 2
    assert meta(page)["rows"][0]["label_type"] == text


def test_rejects_colliding_or_reserved_option_keys():
    for opts in ([("yes", "Yes", "y"), ("no", "No", "Y")], [("yes", "Yes", "j"), ("no", "No", "n")],
                 [("yes", "Yes", "ye"), ("no", "No", "n")]):
        with pytest.raises(ValueError):
            _page(fields=[{"name": "v", "legend": "V", "kind": "radio", "options": opts}],
                  sheet_columns=["packet_id", "label_type", "v"])


def test_rejects_field_missing_from_sheet_or_id_column_as_field():
    with pytest.raises(ValueError):
        _page(sheet_columns=["packet_id", "label_type", "v"])
    with pytest.raises(ValueError):
        _page(sheet_columns=["v", "packet_id", "notes"])
