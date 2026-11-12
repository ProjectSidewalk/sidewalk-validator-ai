"""State test for the round-2 recount without excluded authors' labels (#85).

The recount has two committed states, and this test fails in any state between them:

* pending: ``round2_recount.md`` carries ``<!-- recount-status: pending -->``, ``round2_recount.json`` is absent, and
  the docs quote only the counts before the exclusion;
* done: the marker is gone, ``round2_recount.json`` and ``round2_flags_check.json`` are committed (counts only), the
  "before" half reproduces the committed pool, the known removal is in it (the 11 Seattle items rated unmasked are
  eligible lone AI Disagrees of an excluded author), and the docs quote the JSONs' numbers: the recount tables row by
  row, the pins, the unsampled partition, where the removed labels sit, and the weighted rows of the effect table,
  which are recomputed here from the committed key and sheets; and the exclusion-log table, from the committed
  ``exclusion_changes.csv`` (counts only). Not bound: the split of Chicago's removed labels by exclusion date, whose
  source (the flags, label ids) is not committed.

It never skips because the JSON is missing (that would hide a deleted result); it skips only when the committed pool
is absent, as ``tests/test_gt_rounds.py`` does.
"""
import json
import os
import re

import pytest

from analysis import gt_rounds as gr
from analysis import ai_disagree_relabel as rl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUND2 = os.path.join(ROOT, "data", "relabel-2026-10", "round2")
POOL = os.path.join(ROUND2, "pool.csv.gz")
DOC_DIR = os.path.join(ROOT, "docs", "ai-disagree-relabel-2026-10")
RECOUNT_MD = os.path.join(DOC_DIR, "round2_recount.md")
RECOUNT_JSON = os.path.join(DOC_DIR, "round2_recount.json")
CHECK_JSON = os.path.join(DOC_DIR, "round2_flags_check.json")
MAIN_DOC = os.path.join(ROOT, "docs", "ai-disagree-relabel-2026-10.md")
CHANGES_CSV = os.path.join(ROUND2, "exclusion_changes.csv")
MARKER = "<!-- recount-status: pending -->"
POOL_ROWS = 346_735
BEFORE = {"sidewalk_seattle": (17_284, 8_611), "sidewalk_chicago": (10_769, 3_853)}  # eligible, unsampled
UID = re.compile(r"sidewalk_[a-z0-9_]+:\d|[a-z]+-[a-z]+:\d")

pytestmark = pytest.mark.skipif(not os.path.exists(POOL), reason="committed round-2 pool absent")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _not_sampled_bullet():
    """The main doc's Round 2 "Not sampled" bullet, sliced so number checks cannot match elsewhere in the doc."""
    doc = _read(MAIN_DOC)
    start = doc.index("- **Not sampled.**")
    end = doc.index("\n- **", start + 1)
    return doc[start:end]


def _cells(tree):
    """Every leaf count in a nested dict, keyed by its path."""
    if isinstance(tree, dict):
        return {(k,) + p: v for k, sub in tree.items() for p, v in _cells(sub).items()}
    return {(): tree}


def test_round2_recount_state():
    """Exactly one of (pending marker, committed JSON), then that state's checks. One test, so the pending state
    passes without a skip and the done state's checks cannot be skipped by accident."""
    pending = MARKER in _read(RECOUNT_MD)
    have = os.path.exists(RECOUNT_JSON)
    assert pending != have, ("pending marker and round2_recount.json must not coexist, and one of them must be "
                             "there: remove the marker only when committing the JSON")
    if pending:
        _check_pending()
    else:
        _check_done(json.loads(_read(RECOUNT_JSON)), json.loads(_read(CHECK_JSON)))


def _check_pending():
    """The docs quote only the counts before the exclusion, and no after-number appears in a table row."""
    bullet = _not_sampled_bullet()
    for eligible, unsampled in BEFORE.values():
        assert f"{eligible:,}" in bullet and f"{unsampled:,}" in bullet
    assert "pending" in bullet
    assert not os.path.exists(CHECK_JSON)
    md = _read(RECOUNT_MD)
    for s, (eligible, unsampled) in BEFORE.items():  # the table rows exactly: no count typed in ahead of the JSON
        assert f"| {s} | pending | {eligible:,} -> pending | {unsampled:,} -> pending |" in md, s
    for path in (RECOUNT_MD, MAIN_DOC):
        rows = [ln for ln in _read(path).splitlines() if ln.startswith("|") and re.search(r"->\s*[0-9]", ln)]
        assert rows == [], f"{os.path.basename(path)} has an after-number while the recount is pending: {rows}"


def _check_done(r, chk):
    """Pins, the before half, completeness of the pull, the known removal, counts only, doc sync."""
    # pins, and the "before" half reproduces the committed pool
    for k in ("design", "design_sha256", "pool_sha256", "author_excluded_sha256", "pool_labels_missing_from_flags",
              "before", "after", "excluded_labels", "excluded_eligible_disagrees", "proxy_absent_from_export"):
        assert k in r, k
    assert r["pool_sha256"] == rl.sha256_file(POOL)  # of pool.csv.gz as committed (SHA256SUMS), not the CSV
    assert r["design_sha256"] == rl.sha256_file(gr.DEFAULT_DESIGN)
    design = gr.load_design(gr.DEFAULT_DESIGN)
    assert r["before"] == gr.recount_population(rl.load_pool(POOL), design)
    assert r["author_excluded_sha256"] == chk["author_excluded_sha256"]
    assert chk["checks_passed"] is True
    assert sorted(chk["schemas"]) == sorted(BEFORE)

    # the pull was complete (a failed city would be 146k or 200k labels missing)
    assert r["pool_labels_missing_from_flags"] < 0.01 * POOL_ROWS
    for s in BEFORE:
        assert chk["pool_labels_missing"][s]["share"] < 0.01, s
    assert chk["key_items_flagged"] == chk["unmasked_items"] == 11
    # the two JSONs describe the same flags file: same missing count, and the pool cannot hold more excluded
    # authors' labels than the file has t rows
    assert r["pool_labels_missing_from_flags"] == sum(chk["pool_labels_missing"][s]["n"] for s in BEFORE)
    for s in BEFORE:
        assert r["excluded_labels"][s] <= chk["flagged_t"][s], s

    # the known removal: 10 of the 11 unmasked items are lone Obstacle, 1 lone SurfaceProblem (Seattle)
    assert r["excluded_labels"]["sidewalk_seattle"] >= 11
    before_lone, after_lone = r["before"]["population"]["jon"]["lone"], r["after"]["population"]["jon"]["lone"]
    assert after_lone["Obstacle"] <= before_lone["Obstacle"] - 10
    assert after_lone["SurfaceProblem"] <= before_lone["SurfaceProblem"] - 1
    for s in BEFORE:
        assert 0 <= r["excluded_eligible_disagrees"][s] <= r["excluded_labels"][s]
    before, after = _cells(r["before"]), _cells(r["after"])
    assert before.keys() == after.keys()
    for path, n in after.items():
        assert n <= before[path], path
    # each half adds up: every eligible AI Disagree is in one stratum of its city's rater or in the unsampled rest
    for half in ("before", "after"):
        for rater, s in design["raters"].items():
            in_strata = sum(sum(cells.values()) for cells in r[half]["population"][rater].values())
            assert in_strata + sum(r[half]["unsampled"][s].values()) == r[half]["eligible_disagrees"][s], (half, s)
    for s in BEFORE:
        assert (r["excluded_eligible_disagrees"][s]
                == r["before"]["eligible_disagrees"][s] - r["after"]["eligible_disagrees"][s]), s

    # counts only
    for path in (RECOUNT_JSON, CHECK_JSON):
        text = _read(path)
        assert '"label_id"' not in text and not UID.search(text), path

    # doc sync, anchored on the table row and on the "Not sampled" bullet
    md = _read(RECOUNT_MD)
    bullet = _not_sampled_bullet()
    assert "pending" not in bullet
    # nothing left unfilled: no placeholder section, no pending cell, and the main doc's score sentence is updated
    left = [ln for ln in md.splitlines() if ln.startswith("Pending:") or "-> pending" in ln or "| pending |" in ln]
    assert left == [], left
    assert "**Status: pending" not in md
    assert "once it is run (pending)" not in _read(MAIN_DOC)
    for s, (eligible, _) in BEFORE.items():
        excl = r["excluded_labels"][s]
        after_n = r["after"]["eligible_disagrees"][s]
        assert f"| {s} | {excl:,} | {eligible:,} -> {after_n:,} |" in md, s
        assert f"{after_n:,}" in bullet and f"{sum(r['after']['unsampled'][s].values()):,}" in bullet, s
    _check_doc_numbers(r, chk, design, md, bullet)
    _check_exclusion_log(md)
    _check_weighted_rows(r, md)


def _flat(text):
    """Text with every run of whitespace as one space, so a check does not depend on where a line wraps."""
    return " ".join(text.split())


def _check_doc_numbers(r, chk, design, md, bullet):
    """Every count the two docs quote from the two JSONs: the `recount` tables row by row, the pins, the unsampled
    partition, and where the removed labels sit."""
    # line by line: the JSON is written with sorted keys, so the rows come back in another order than `recount` printed
    md_lines = set(md.splitlines())
    missing = [ln for ln in gr.recount_markdown(r, design).splitlines() if ln and ln not in md_lines]
    assert missing == [], f"recount table lines in round2_recount.md that are not the JSON's: {missing}"
    main = _read(MAIN_DOC)
    flags_sha = r["author_excluded_sha256"]
    assert f"`{flags_sha}`" in md and f"`{flags_sha}`" in main
    for sha in (r["pool_sha256"], r["design_sha256"], rl.sha256_file(CHECK_JSON)):
        assert f"`{sha}`" in md, sha
    for s in BEFORE:
        assert chk["rows"][s] == chk["flagged_t"][s] + chk["flagged_f"][s], s
        assert r["excluded_labels"][s] <= chk["pool_labels"][s] <= chk["rows"][s], s
    sea, chi = "sidewalk_seattle", "sidewalk_chicago"
    assert (f"Seattle {chk['rows'][sea]:,} rows, {chk['flagged_t'][sea]:,} `t`; "
            f"Chicago {chk['rows'][chi]:,} rows, {chk['flagged_t'][chi]:,} `t`") in md

    def part(half, s):
        u = r[half]["unsampled"][s]
        return (f"{sum(u.values()):,} of {r[half]['eligible_disagrees'][s]:,}",
                f"({u['one_decisive']:,} / {u['unsure_only']:,} / {u['tie']:,})")

    flat_md, flat_bullet = _flat(md), _flat(bullet)
    for name, s in (("Seattle", sea), ("Chicago", chi)):
        (a_of, a_parts), (b_of, b_parts) = part("after", s), part("before", s)
        assert f"- {name}: {a_of} eligible {a_parts}, before {b_of} {b_parts}." in flat_md, s
        assert f"{b_of} {b_parts}" in flat_bullet, s  # the counts `build` printed
    ua, uc = r["after"]["unsampled"][sea], r["after"]["unsampled"][chi]
    assert (f"({ua['one_decisive']:,} with one decisive vote, {ua['unsure_only']:,} with only Unsure votes, "
            f"{ua['tie']:,} ties)") in flat_bullet
    assert f"({uc['one_decisive']:,} / {uc['unsure_only']:,} / {uc['tie']:,})" in flat_bullet

    # where the removed labels sit: per stratum and, for Seattle's lone stratum, per type
    def removed(rater, s):
        pb, pa = r["before"]["population"][rater], r["after"]["population"][rater]
        by = {st: {t: pb[st][t] - pa[st][t] for t in pa[st]} for st in pa}
        return by, (sum(r["before"]["unsampled"][s].values()) - sum(r["after"]["unsampled"][s].values()))

    by, uns = removed("jon", sea)
    n_sea, n_chi = r["excluded_eligible_disagrees"][sea], r["excluded_eligible_disagrees"][chi]
    lone = sum(by["lone"].values())
    types = ", ".join(f"{t} {n}" for t, n in sorted(by["lone"].items(), key=lambda kv: -kv[1]) if n)
    assert f"{n_sea} are eligible AI Disagrees: {lone} in `lone` ({types}) and {uns} unsampled" in flat_md
    assert sum(sum(c.values()) for st, c in by.items() if st != "lone") == 0  # "No Seattle label leaves ..."
    assert f"removes {n_sea} Seattle and {n_chi} Chicago AI Disagrees, {lone} of the Seattle ones from `lone`" \
        in flat_bullet
    by, uns = removed("mikey", chi)
    assert (f"{n_chi} are eligible AI Disagrees: {sum(by['margin_incorrect'].values())} `margin_incorrect`, "
            f"{sum(by['lone'].values())} `lone`, {sum(by['light_voted'].values())} `light_voted` and {uns} "
            "unsampled") in flat_md


def _check_exclusion_log(md):
    """The exclusion-log table and the author-mismatch counts, from the committed ``exclusion_changes.csv`` as pulled
    (its sha256 is pinned in the doc): one table row per logged toggle, "none logged" for a city without one, and
    nothing but counts, a day, a schema and true/false in the file."""
    assert f"`{rl.sha256_file(CHANGES_CSV)}`" in md
    changes, mismatch = {s: [] for s in BEFORE}, {}
    for ln in _read(CHANGES_CSV).splitlines():
        f = ln.split(",")
        if f[0] == "change":
            assert re.fullmatch(r"change,sidewalk_[a-z_]+,\d{4}-\d\d-\d\d,(true|false),\d+,\d+", ln), ln
            changes[f[1]].append((f[2], "excluded" if f[3] == "true" else "not excluded", int(f[4]), int(f[5])))
        else:
            assert re.fullmatch(r"author_mismatch,sidewalk_[a-z_]+,\d+", ln), ln
            mismatch[f[1]] = int(f[2])
    assert sorted(mismatch) == sorted(BEFORE)
    rows = [ln for ln in md.splitlines() if re.match(r"\| sidewalk_[a-z_]+ \| (none logged|\d{4}-\d\d-\d\d) \|", ln)]
    want = []
    for s in ("sidewalk_seattle", "sidewalk_chicago"):
        want += [f"| {s} | {day} | {value} | {authors} | {labels:,} |" for day, value, authors, labels in changes[s]]
        if not changes[s]:
            want.append(f"| {s} | none logged | | | |")
    assert sorted(rows) == sorted(want), (rows, want)
    n = sum(len(v) for v in changes.values())
    word = {0: "no", 1: "one", 2: "two", 3: "three"}.get(n, str(n))
    assert f"returned {word} change row{'' if n == 1 else 's'} and both `author_mismatch` rows" in _flat(md)
    assert (f"Seattle {mismatch['sidewalk_seattle']:,}, Chicago {mismatch['sidewalk_chicago']:,}, whole schema"
            in _flat(md))


def _check_weighted_rows(r, md):
    """The effect table, recomputed: ``score_table`` on the committed key and sheets (unmasked items left out), once
    with each half's population. Binds the decided n, both populations, the population in cells that carry weight,
    and both weighted rates with their intervals; and the main doc's sentence on what moved."""
    import pandas as pd
    pre = os.path.join(ROUND2, "pre-consensus-2026-11-02")
    key = pd.read_csv(os.path.join(ROUND2, "key", "key.csv"), dtype={"tags": str})
    unmasked = gr.unmasked_ids(pd.read_csv(os.path.join(pre, "unmasked.csv"), dtype=str), key)
    moved = []
    for rater in ("jon", "mikey"):
        m = rl.unmask(gr.rater_items(key, rater), gr._read_sheet(os.path.join(pre, f"votes_{rater}.csv")))
        own = m[(m.rater == rater) & ~m.packet_id.isin(unmasked)]
        decided = own[own.verdict.isin(["correct", "incorrect"])]
        halves = {h: gr.score_table(own, r[h]["population"][rater]) for h in ("before", "after")}
        for st in gr.DISAGREE_STRATA:
            cells, pcts = [], []
            for h in ("before", "after"):
                pop = r[h]["population"][rater][st]
                row = halves[h][(halves[h].stratum == st) & (halves[h].label_type == "all")].iloc[0]
                small = {x.split("(")[0] for x in row.w_excluded.split(", ") if x}
                assert all(pop[t] > 0 for t in small), (rater, st, h)  # a no-weight cell is still named
                cells.append((sum(pop.values()), sum(n for t, n in pop.items() if t not in small)))
                pcts.append(rl._pct(row.w_rate, row.w_lo, row.w_hi))
            n = int((decided.stratum == st).sum())
            want = (f"| {st} | {n} | {cells[0][0]:,} -> {cells[1][0]:,} | {cells[0][1]:,} -> {cells[1][1]:,} | "
                    f"{pcts[0]} | {pcts[1]} |")
            rows = [ln for ln in md.splitlines() if ln.startswith(f"| {rater} (") and f"| {st} |" in ln]
            assert len(rows) == 1 and rows[0].endswith(want), (rater, st, want, rows)
            if pcts[0] != pcts[1]:
                moved.append((rater, st, pcts[0], pcts[1]))
    # exactly the rows the docs say move, with the values they quote
    assert [(a, b) for a, b, _, _ in moved] == [("jon", "lone"), ("mikey", "light_voted")], moved
    (_, _, b0, a0), (_, _, b1, a1) = moved
    flat_md, flat_main = _flat(md), _flat(_read(MAIN_DOC))
    assert f"Seattle `lone` from {b0} to {a0}" in flat_md
    lo = [x.split("[")[1].split(",")[0] for x in (b1, a1)]
    assert b1.split("[")[0] == a1.split("[")[0] and b1.split(", ")[1] == a1.split(", ")[1]  # only the lower bound
    assert f"the lower bound of Chicago `light_voted` from {lo[0]} to {lo[1]}" in flat_md
    assert (f"(Seattle lone {b0.split(' ')[0]} -> {a0.split(' ')[0]}; the lower bound of Chicago light_voted "
            f"{lo[0]} -> {lo[1]}; no other printed value moves)") in flat_main
