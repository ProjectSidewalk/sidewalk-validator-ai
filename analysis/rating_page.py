"""A self-contained HTML page for rating image items, one card per item, that exports a filled CSV sheet.

Modelled on RampNet's GT-check galleries (``benchmark/*/gallery.html`` there): each card has radio answers and a
notes box; answers are kept in the browser per rater id as you go; Export writes a CSV with the same header and row
order as the blank sheet, so the analysis that reads the sheet runs on it unchanged; Import reads such a CSV back
(to resume in another browser). Inline CSS and vanilla JS, no network, no libraries.

Keyboard: each radio option may carry a one-character key; it answers the card that has focus or, when focus is
outside every card, the topmost card on screen. J / K move to the next / previous card. Keys do nothing inside a
text field.

An optional **gate** keeps the cards hidden until the rater loads a CSV in which every item has an allowed value
in one column (e.g. a filled pass-1 sheet before pass 2). It prevents opening a later pass first by accident; it is
not a security boundary, since the images sit next to the page.

Example::

    page = render_page(
        title="Rate crops", heading="Rate crops", intro_html="<p>Is the label correct?</p>",
        items=[{"id": "p1a2b3c", "image": "items/001_p1a2b3c.jpg", "alt": "Item 1: curb ramp",
                "caption_html": "1. p1a2b3c &middot; curb ramp"}],
        fields=[{"name": "my_verdict", "legend": "Correct label?", "kind": "radio",
                 "options": [("correct", "Correct", "c"), ("incorrect", "Incorrect", "i")]},
                {"name": "notes", "legend": "Notes (optional)", "kind": "text"}],
        sheet_columns=["packet_id", "label_type", "my_verdict", "notes"],
        fixed={"p1a2b3c": {"label_type": "CurbRamp"}},
        storage_key="vlm16_pass1", export_name="pass1_sheet")
"""
import html
import json

CSS = """
:root { color-scheme: light dark; --bg:#ffffff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --focus:#0969da;
  --panel:#f6f8fa; --done:#1a7f37; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d;
  --focus:#4493f8; --panel:#161b22; --done:#3fb950; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 16px 64px;
  max-width:1200px; }
h1 { font-size:22px; } h2 { font-size:16px; margin:0 0 8px; }
.intro { max-width:75ch; } .muted { color:var(--muted); font-size:13px; }
.bar { position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1;
  display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }
.card { border-top:1px solid var(--line); padding:14px 0; scroll-margin-top:64px; }
.card:focus { outline:none; } .card:focus-visible h2, .card.active h2 { text-decoration:underline; }
.card.done h2::after { content:" \\2713"; color:var(--done); }
.row { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start; }
.pic { flex:0 1 640px; min-width:0; overflow:auto; }
.pic button { all:unset; cursor:zoom-in; display:block; }
.pic button:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
.pic img { max-width:100%; height:auto; display:block; }
.pic button[aria-pressed=true] { cursor:zoom-out; } .pic button[aria-pressed=true] img { max-width:none; }
.form { flex:1 1 320px; min-width:0; }
.extra { margin:0 0 8px; }
fieldset { border:1px solid var(--line); border-radius:6px; background:var(--panel); margin:0 0 8px;
  padding:8px 10px; }
legend { font-weight:600; padding:0 4px; }
.opts { display:flex; flex-wrap:wrap; gap:4px 18px; } .opts label { cursor:pointer; padding:2px 0; }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-radius:3px; padding:0 4px; }
textarea { display:block; width:100%; box-sizing:border-box; font:inherit; color:var(--fg); background:var(--bg);
  border:1px solid var(--line); border-radius:4px; }
input[type=text] { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line);
  border-radius:4px; padding:4px 6px; }
:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
button { font:inherit; padding:6px 12px; }
.gate { border:1px solid var(--line); border-radius:6px; background:var(--panel); padding:12px; margin:16px 0; }
[hidden] { display:none !important; }
"""

JS = r"""
const META = JSON.parse(document.getElementById("meta").textContent);
const RADIOS = META.fields.filter(f => f.kind === "radio");
const KEYS = {};
for (const f of RADIOS) for (const [v, , k] of f.options) if (k) KEYS[k.toLowerCase()] = [f.name, v];
const $ = id => document.getElementById(id);
const cards = Array.from(document.querySelectorAll(".card"));
const raterBox = $("rater");
let answers = {};

function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); }
                       catch (e) { return null; } }
function rater() { return raterBox.value.trim().toLowerCase(); }
function slot() { return META.storage + ":" + (rater() || "default"); }
function say(text) { $("msg").textContent = text; }

function load() {
  try { answers = JSON.parse(store(slot()) || "{}") || {}; } catch (e) { answers = {}; }
  for (const c of cards) paint(c);
  count();
  if (META.gate) gateCheck();
}
function save() { store(slot(), JSON.stringify(answers)); }

function paint(card) {
  const a = answers[card.dataset.id] || {};
  for (const f of META.fields) {
    if (f.kind === "radio") {
      for (const r of card.querySelectorAll(`input[name="${CSS.escape(card.dataset.id + "__" + f.name)}"]`)) r.checked = r.value === a[f.name];
    } else {
      card.querySelector(`textarea[data-field="${CSS.escape(f.name)}"]`).value = a[f.name] || "";
    }
  }
  card.classList.toggle("done", complete(card.dataset.id));
}
function complete(id) { const a = answers[id] || {}; return RADIOS.every(f => a[f.name]); }
function count() {
  const n = cards.filter(c => complete(c.dataset.id)).length;
  $("count").textContent = `${n} of ${cards.length} answered`;
}
function set(card, field, value) {
  const id = card.dataset.id;
  answers[id] = Object.assign({}, answers[id], {[field]: value});
  save(); paint(card); count();
}

document.addEventListener("change", ev => {
  const card = ev.target.closest(".card");
  if (card && ev.target.type === "radio") set(card, ev.target.dataset.field, ev.target.value);
});
document.addEventListener("input", ev => {
  const card = ev.target.closest(".card");
  if (card && ev.target.tagName === "TEXTAREA") set(card, ev.target.dataset.field, ev.target.value);
});
raterBox.addEventListener("change", () => { store(META.storage + ":last_rater", rater()); load(); });
for (const b of document.querySelectorAll(".pic button"))
  b.addEventListener("click", () => b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true"));

function visibleCards() { return $("cards").hidden ? [] : cards; }
function current() {
  const f = document.activeElement && document.activeElement.closest && document.activeElement.closest(".card");
  if (f) return f;
  const top = document.querySelector(".bar").getBoundingClientRect().bottom;
  return visibleCards().find(c => c.getBoundingClientRect().bottom > top + 40);
}
function go(card) {
  if (!card) return;
  for (const c of cards) c.classList.toggle("active", c === card);
  card.focus({preventScroll: true});
  card.scrollIntoView({block: "start", behavior: "smooth"});
}
document.addEventListener("keydown", ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const t = ev.target;
  if (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text")) return;
  const card = current();
  if (!card) return;
  const k = ev.key.toLowerCase(), list = visibleCards(), i = list.indexOf(card);
  if (k === "j") { go(list[i + 1]); ev.preventDefault(); return; }
  if (k === "k") { go(list[i - 1]); ev.preventDefault(); return; }
  if (!KEYS[k]) return;
  ev.preventDefault();
  const [field, value] = KEYS[k];
  set(card, field, value);
  if (complete(card.dataset.id)) go(list[i + 1]);
  else go(card);
});
$("next").addEventListener("click", () => {
  const c = visibleCards().find(c => !complete(c.dataset.id));
  if (c) go(c); else say("Every card is answered.");
});

// ---- CSV ---------------------------------------------------------------------------------------------------------
function csvCell(v) { v = v == null ? "" : String(v); return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function parseCsv(text) {
  const rows = []; let row = [], cell = "", q = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') q = false;
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); rows.push(row); row = []; cell = "";
    } else cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  const head = rows.shift() || [];
  return rows.filter(r => r.some(c => c !== "")).map(r => Object.fromEntries(head.map((h, j) => [h.trim(), r[j] || ""])));
}
function readFile(input, then) {
  const f = input.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => { try { then(parseCsv(r.result)); } catch (e) { say("Could not read " + f.name + ": " + e.message); } input.value = ""; };
  r.readAsText(f);
}
const ID = META.columns[0];

$("export").addEventListener("click", () => {
  const lines = [META.columns.map(csvCell).join(",")];
  for (const row of META.rows) {
    const a = answers[row[ID]] || {};
    lines.push(META.columns.map(c => csvCell(c in row ? row[c] : (a[c] || "").trim())).join(","));
  }
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = META.export + (rater() ? "__" + rater() : "") + ".csv";
  document.body.appendChild(link); link.click(); link.remove();
  const open = cards.length - cards.filter(c => complete(c.dataset.id)).length;
  say(`Exported ${link.download}` + (open ? ` (${open} cards still unanswered)` : ""));
});
$("importfile").addEventListener("change", ev => readFile(ev.target, rows => {
  const ids = new Set(META.rows.map(r => r[ID]));
  let n = 0;
  for (const r of rows) {
    if (!ids.has(r[ID])) continue;
    const a = {};
    for (const f of META.fields) if (r[f.name]) a[f.name] = r[f.name].trim();
    answers[r[ID]] = Object.assign({}, answers[r[ID]], a); n++;
  }
  save(); load(); say(`Imported answers for ${n} cards.`);
}));

// ---- gate --------------------------------------------------------------------------------------------------------
function gateCheck() {
  const open = store(META.storage + ":gate:" + (rater() || "default")) === "ok";
  $("cards").hidden = !open; $("gate").hidden = open;
}
if (META.gate) $("gatefile").addEventListener("change", ev => readFile(ev.target, rows => {
  const g = META.gate, got = {};
  for (const r of rows) got[r[ID]] = (r[g.column] || "").trim().toLowerCase();
  const missing = META.rows.map(r => r[ID]).filter(id => !g.values.includes(got[id]));
  if (missing.length) { say(`That sheet is not complete: ${missing.length} of ${META.rows.length} items have no ${g.column}.`); return; }
  store(META.storage + ":gate:" + (rater() || "default"), "ok");
  gateCheck(); say("Pass-1 sheet checked. The cards are open.");
}));

const params = new URLSearchParams(location.search);
raterBox.value = params.get("rater") || store(META.storage + ":last_rater") || "";
load();
"""


def _e(s):
    return html.escape(str(s), quote=True)


def _card(item, fields):
    """One ``<section class="card">``: the image (click toggles 1:1) and the answer form."""
    iid = _e(item["id"])
    parts = [f'<section class="card" data-id="{iid}" tabindex="-1" aria-labelledby="h_{iid}">',
             f'<h2 id="h_{iid}">{item["caption_html"]}</h2><div class="row">',
             f'<div class="pic"><button type="button" aria-pressed="false" aria-label="Toggle full size for '
             f'{_e(item["alt"])}"><img src="{_e(item["image"])}" alt="{_e(item["alt"])}" loading="lazy"></button>'
             f'</div><div class="form">']
    if item.get("extra_html"):
        parts.append(f'<div class="extra">{item["extra_html"]}</div>')
    for f in fields:
        if f["kind"] == "radio":
            opts = "".join(
                f'<label><input type="radio" name="{iid}__{_e(f["name"])}" value="{_e(v)}" '
                f'data-field="{_e(f["name"])}"> {_e(label)}{f" <kbd>{_e(k.upper())}</kbd>" if k else ""}</label>'
                for v, label, k in f["options"])
            parts.append(f'<fieldset><legend>{_e(f["legend"])}</legend><div class="opts">{opts}</div></fieldset>')
        else:
            tid = f'{iid}__{_e(f["name"])}'
            parts.append(f'<label for="{tid}">{_e(f["legend"])}</label>'
                         f'<textarea id="{tid}" rows="2" data-field="{_e(f["name"])}"></textarea>')
    parts.append("</div></div></section>")
    return "".join(parts)


def render_page(*, title, heading, intro_html, items, fields, sheet_columns, fixed, storage_key, export_name,
                gate=None):
    """The page as one HTML string.

    ``items``: dicts with ``id``, ``image`` (path relative to the page), ``alt``, ``caption_html`` and optional
    ``extra_html`` (both trusted HTML), in sheet order. ``fields``: dicts with ``name`` (a sheet column), ``legend``,
    ``kind`` (``"radio"`` with ``options`` as ``(value, label, key)`` triples, or ``"text"``). ``sheet_columns``:
    the CSV header; the first column is the item id. ``fixed``: ``{id: {column: value}}`` for the columns that are
    not answers, written back unchanged on export. ``gate``: optional ``{"column", "values", "html"}``: the cards
    stay hidden until a CSV is loaded in which every item's ``column`` holds one of ``values``.

    >>> p = render_page(title="t", heading="h", intro_html="", items=[{"id": "a", "image": "a.jpg", "alt": "A",
    ...     "caption_html": "1"}], fields=[{"name": "v", "legend": "V", "kind": "radio",
    ...     "options": [("yes", "Yes", "y")]}], sheet_columns=["id", "v"], fixed={"a": {}}, storage_key="k",
    ...     export_name="sheet")
    >>> '"columns": ["id", "v"]' in p and 'value="yes"' in p
    True
    """
    names = {f["name"] for f in fields}
    if sheet_columns[0] in names or not names <= set(sheet_columns):
        raise ValueError("sheet_columns must start with the id column and include every field")
    rows = []
    for it in items:
        row = {sheet_columns[0]: it["id"]}
        for c in sheet_columns[1:]:
            if c not in names:
                row[c] = str(fixed.get(it["id"], {}).get(c, ""))
        rows.append(row)
    meta = {"storage": storage_key, "export": export_name, "columns": list(sheet_columns), "rows": rows,
            "fields": [{"name": f["name"], "kind": f["kind"], "options": [list(o) for o in f.get("options", [])]}
                       for f in fields],
            "gate": {"column": gate["column"], "values": list(gate["values"])} if gate else None}
    meta_js = json.dumps(meta, ensure_ascii=False).replace("</", "<\\/")
    gate_html = ""
    if gate:
        gate_html = (f'<div class="gate" id="gate">{gate["html"]}<p><label for="gatefile">Filled sheet</label> '
                     f'<input type="file" id="gatefile" accept=".csv,text/csv"></p></div>')
    return "".join([
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{_e(heading)}</h1><div class=\"intro\">{intro_html}</div>",
        '<p class="muted">Answers are saved in this browser per rater id as you go. <strong>Export</strong> writes '
        f'the filled sheet (<code>{_e(export_name)}__&lt;rater&gt;.csv</code>); <strong>Import</strong> loads one '
        'back. Keys: the letter beside an answer sets it on the current card (the focused card, or the top one on '
        'screen) and moves on once the card is complete; <kbd>J</kbd> / <kbd>K</kbd> next / previous card. Click an '
        'image for full size.</p>',
        '<div class="bar"><label for="rater">Rater id</label>',
        '<input id="rater" type="text" size="10" autocomplete="off" spellcheck="false" aria-describedby="msg">',
        '<button type="button" id="export">Export sheet CSV</button>',
        '<label>Import <input type="file" id="importfile" accept=".csv,text/csv"></label>',
        '<button type="button" id="next">Next unanswered</button>',
        '<span id="count" aria-live="polite"></span> <span id="msg" role="status"></span></div>',
        gate_html,
        f'<div id="cards"{" hidden" if gate else ""}>',
        "".join(_card(it, fields) for it in items),
        f'</div><script id="meta" type="application/json">{meta_js}</script><script>{JS}</script></body></html>\n',
    ])
