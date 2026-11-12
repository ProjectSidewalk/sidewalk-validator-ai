"""State test for the round-2 recount without excluded authors' labels (#85).

The recount has two committed states, and this test fails in any state between them:

* pending: ``round2_recount.md`` carries ``<!-- recount-status: pending -->``, ``round2_recount.json`` is absent, and
  the docs quote only the counts before the exclusion;
* done: the marker is gone, ``round2_recount.json`` and ``round2_flags_check.json`` are committed (counts only), the
  "before" half reproduces the committed pool, the known removal is in it (the 11 Seattle items rated unmasked are
  eligible lone AI Disagrees of an excluded author), and the docs quote the "after" numbers.

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

    # the pull was complete (a failed city would be 146k or 200k labels missing)
    assert r["pool_labels_missing_from_flags"] < 0.01 * POOL_ROWS
    for s in BEFORE:
        assert chk["pool_labels_missing"][s]["share"] < 0.01, s
    assert chk["key_items_flagged"] == chk["unmasked_items"] == 11

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

    # counts only
    for path in (RECOUNT_JSON, CHECK_JSON):
        text = _read(path)
        assert '"label_id"' not in text and not UID.search(text), path

    # doc sync, anchored on the table row and on the "Not sampled" bullet
    md = _read(RECOUNT_MD)
    bullet = _not_sampled_bullet()
    assert "pending" not in bullet
    for s, (eligible, _) in BEFORE.items():
        excl = r["excluded_labels"][s]
        after_n = r["after"]["eligible_disagrees"][s]
        assert f"| {s} | {excl:,} | {eligible:,} -> {after_n:,} |" in md, s
        assert f"{after_n:,}" in bullet and f"{sum(r['after']['unsampled'][s].values()):,}" in bullet, s
