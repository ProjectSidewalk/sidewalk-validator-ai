"""Browser tests for analysis/rating_page.py's JS (headless Chromium via Playwright; skipped when it is absent,
as in CI). Run locally with a Python that has playwright and pytest."""
import csv
import io

import pytest

from analysis import rating_page as rp

sync_api = pytest.importorskip("playwright.sync_api")

IDS = [f"p{i:06x}" for i in range(1, 13)]
COLUMNS = ["packet_id", "label_type", "v", "w", "notes"]
FIELDS = [{"name": "v", "legend": "V?", "kind": "radio", "options": [("yes", "Yes", "y"), ("no", "No", "n")]},
          {"name": "w", "legend": "W?", "kind": "radio", "options": [("a", "A", "1"), ("b", "B", "0")]},
          {"name": "notes", "legend": "Notes", "kind": "text"}]


def write_page(tmp_path, gate=None):
    items = [{"id": i, "image": f"items/{n:03d}.jpg", "alt": f"Item {n}", "caption_html": f"{n}. {i}"}
             for n, i in enumerate(IDS, 1)]
    page = rp.render_page(title="T", heading="H", intro_html="<p>i</p>", items=items, fields=FIELDS,
                          sheet_columns=COLUMNS, fixed={i: {"label_type": f'T "{n}", x'} for n, i in enumerate(IDS)},
                          storage_key="test", export_name="sheet", gate=gate)
    path = tmp_path / "rate.html"
    path.write_text(page)
    return path.as_uri()


@pytest.fixture()
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # browsers not installed
            pytest.skip(f"chromium unavailable: {e}")
        yield b
        b.close()


def open_page(browser, url, rater="r1"):
    ctx = browser.new_context(accept_downloads=True, viewport={"width": 1000, "height": 500})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url)
    if rater:
        pg.fill("#rater", rater)
        pg.press("#rater", "Tab")
    pg.click("h1")
    return pg, errors


def export(pg, tmp_path):
    with pg.expect_download() as d:
        pg.click("#export")
    out = tmp_path / d.value.suggested_filename
    d.value.save_as(out)
    return out


def test_keys_need_a_rater_id_and_answers_persist_per_rater(browser, tmp_path):
    url = write_page(tmp_path)
    pg, errors = open_page(browser, url, rater=None)
    pg.keyboard.press("y")
    assert pg.text_content("#count").startswith("0 of 12") and "rater id" in pg.text_content("#msg")
    pg.fill("#rater", "r1")
    pg.press("#rater", "Tab")
    pg.click("h1")
    for k in "y1n0":
        pg.keyboard.press(k)
    assert pg.text_content("#count").startswith("2 of 12")
    pg.reload()
    assert pg.input_value("#rater") == "r1" and pg.text_content("#count").startswith("2 of 12")
    pg.fill("#rater", "r2")
    pg.press("#rater", "Tab")
    assert pg.text_content("#count").startswith("0 of 12")
    assert errors == []


def test_key_repeat_is_ignored(browser, tmp_path):
    pg, _ = open_page(browser, write_page(tmp_path))
    for _ in range(4):
        pg.evaluate("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'y', repeat: true, bubbles: true}))")
    pg.keyboard.press("y")
    assert pg.eval_on_selector_all("input[value=yes]:checked", "els => els.length") == 1


def test_keys_act_on_a_visible_card_after_scrolling_away_from_the_focused_one(browser, tmp_path):
    pg, _ = open_page(browser, write_page(tmp_path))
    pg.keyboard.press("y")
    pg.keyboard.press("1")                                      # card 1 done; focus moves to card 2
    pg.wait_for_timeout(600)
    pg.mouse.wheel(0, 5000)
    pg.wait_for_timeout(600)
    pg.keyboard.press("n")
    checked = pg.eval_on_selector_all("input[value=no]:checked", "els => els.map(e => e.name)")
    assert len(checked) == 1 and checked[0] != f"{IDS[1]}__v"
    box = pg.locator(f'.card[data-id="{checked[0].split("__")[0]}"]').bounding_box()
    assert box["y"] < 500 and box["y"] + box["height"] > 0


def test_export_matches_the_sheet_format_and_import_reads_it_back(browser, tmp_path):
    pg, errors = open_page(browser, write_page(tmp_path))
    pg.keyboard.press("y")
    pg.keyboard.press("0")
    pg.fill(f'textarea[data-field="notes"] >> nth=0', 'say "hi", then\nleave')
    rows = list(csv.reader(io.StringIO(export(pg, tmp_path).read_text())))
    assert rows[0] == COLUMNS and [r[0] for r in rows[1:]] == IDS
    assert rows[1] == [IDS[0], 'T "0", x', "yes", "b", 'say "hi", then\nleave']
    assert all(r[2:] == ["", "", ""] for r in rows[2:])
    path = tmp_path / "sheet__r1.csv"
    # import into a new rater: case and spaces normalised, values outside the rubric and unknown ids skipped
    text = path.read_text().replace(",yes,b,", ", YES ,maybe,") + "p999999,x,yes,a,\n"
    path.write_text(text)
    pg.fill("#rater", "r2")
    pg.press("#rater", "Tab")
    pg.set_input_files("#importfile", str(path))
    pg.wait_for_timeout(300)
    assert "1 values outside the rubric" in pg.text_content("#msg")
    assert pg.is_checked(f'input[name="{IDS[0]}__v"][value=yes]')
    assert not pg.is_checked(f'input[name="{IDS[0]}__w"][value=b]')
    assert pg.input_value('textarea[data-field="notes"] >> nth=0') == 'say "hi", then\nleave'
    assert errors == []


def test_gate_opens_only_on_a_complete_sheet(browser, tmp_path):
    url = write_page(tmp_path, gate={"column": "verdict", "values": ["ok"], "html": "<p>load</p>"})
    pg, errors = open_page(browser, url)
    assert pg.is_hidden("#cards")

    def load(rows, header=("packet_id", "verdict"), name="g.csv"):
        f = tmp_path / name
        f.write_text("﻿" + ",".join(header) + "\r\n" + "".join(f"{a},{b}\r\n" for a, b in rows))
        pg.set_input_files("#gatefile", str(f))
        pg.wait_for_timeout(300)
        return pg.text_content("#msg")

    assert "columns" in load([(i, "ok") for i in IDS], header=("packet_id", "other"))
    assert "1 are not in it" in load([(i, "ok") for i in IDS[1:]])
    assert "1 have no valid" in load([(i, "ok") for i in IDS[:-1]] + [(IDS[-1], "")])
    assert pg.is_hidden("#cards")
    assert "open" in load([(i, " OK ") for i in IDS])
    assert not pg.is_hidden("#cards")
    pg.fill("#rater", "r2")
    pg.press("#rater", "Tab")
    assert pg.is_hidden("#cards")                               # the gate is per rater
    assert errors == []
