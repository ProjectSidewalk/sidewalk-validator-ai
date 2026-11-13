"""Acceptance check for an author-exclusion pull (issue #85), run locally before the flags are used.

``analysis/pull_author_excluded.sh`` prints ``schema,label_id,author_excluded`` for every live label of a round's
schemas. ``gt_rounds.py recount`` / ``score --author-excluded`` join that file onto the round's pinned pool and treat
a pool label missing from it as *not* excluded, so a partial or garbled pull would silently give wrong weights. This
script refuses such a file. Checks, in order (1-3 on the raw lines, before any parsing library sees them):

1. the file is UTF-8 and its header is exactly ``schema,label_id,author_excluded``;
2. every row has exactly 3 fields (a stray ``COPY n`` tag or a truncated last line does not);
3. the third field is ``t`` or ``f`` and ``label_id`` is a non-negative integer of at most 18 digits (it fits
   int64). Lines end in a bare newline, as psql writes them: a carriage return fails check 1 (header) or 3 (row);
4. the set of schemas is exactly ``--schemas`` (no city missing, none extra);
5. ``(schema, label_id)`` is unique;
6. every schema of the pool is in ``--schemas`` (``--schemas`` is what the pull was asked for, so only the pool can
   show that a whole city was left out), and per schema the share of the pool's labels missing from the flags is
   under ``--max-missing`` (a city whose query failed shows up as 100%);
7. the whole key: every key item is joined onto the flags on ``(schema, label_id)`` (one-to-one), and the key items
   flagged ``t`` must be exactly the items of ``--unmasked`` (rated unmasked because their author is excluded). A
   key item missing from the flags is counted; an unmasked item missing is a failure.

Every failure is named on stderr (label references as ``seattle:216613``); the exit status is 0 only if all pass.
``--out`` (written only when all pass) holds counts only, no label ids: rows, ``t`` and ``f`` per schema, pool labels
missing per schema, key items flagged and missing, and the flags' sha256.

Usage::

    python analysis/check_author_excluded.py \\
        --flags ~/Data/sidewalk-validator-ai-relabel-2026-10/round2/author_excluded.csv.tmp \\
        --pool data/relabel-2026-10/round2/pool.csv.gz --key data/relabel-2026-10/round2/key/key.csv \\
        --unmasked data/relabel-2026-10/round2/pre-consensus-2026-11-02/unmasked.csv \\
        --schemas sidewalk_seattle sidewalk_chicago --out docs/ai-disagree-relabel-2026-10/round2_flags_check.json

Normally called by ``analysis/round2_flags_pull.sh``. Needs only pandas (no repo imports).
"""
import argparse
import hashlib
import json
import re
import sys

import pandas as pd

HEADER = "schema,label_id,author_excluded"
_INT = re.compile(r"[0-9]{1,18}")  # at most 18 digits: always fits int64
_SHOW = 5  # bad lines / label refs named per failure before "... and N more"


def short_ref(schema, label_id):
    """``seattle:216613`` for ``("sidewalk_seattle", 216613)``: the city-qualified way to name a label.

    >>> short_ref("sidewalk_seattle", 216613)
    'seattle:216613'
    """
    return f"{schema.removeprefix('sidewalk_')}:{int(label_id)}"


def _named(items):
    items = sorted(items)
    more = f" ... and {len(items) - _SHOW} more" if len(items) > _SHOW else ""
    return ", ".join(items[:_SHOW]) + more


def _lines(numbers):
    more = f" ... and {len(numbers) - _SHOW} more" if len(numbers) > _SHOW else ""
    return ", ".join(str(n) for n in numbers[:_SHOW]) + more


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_flags_lines(path):
    """``(rows, failures)``: the data rows as ``(schema, label_id, value)`` tuples if checks 1-3 pass, else ``None``
    and the failures. A final newline is allowed; any other empty line counts as a row with the wrong field count.
    Nothing is stripped: a carriage return, a space or a quote makes its line fail.

    >>> import io, tempfile, os
    >>> d = tempfile.mkdtemp(); p = os.path.join(d, "f.csv")
    >>> _ = open(p, "w").write("schema,label_id,author_excluded\\nsidewalk_a,1,t\\nCOPY 1\\n")
    >>> read_flags_lines(p)[1]
    ['check 2: 1 row(s) without exactly 3 fields: line 3']
    """
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, [f"check 1: the file is not UTF-8 text ({e.reason} at byte {e.start})"]
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != HEADER:
        got = repr(lines[0]) if lines else "an empty file"
        return None, [f"check 1: header is {got}, expected {HEADER!r}"]
    bad_fields, bad_values, rows = [], [], []
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        if len(parts) != 3:
            bad_fields.append(n)
            continue
        schema, label_id, value = parts
        if value not in ("t", "f") or not _INT.fullmatch(label_id):
            bad_values.append(n)
            continue
        rows.append((schema, int(label_id), value))
    failures = []
    if bad_fields:
        failures.append(f"check 2: {len(bad_fields)} row(s) without exactly 3 fields: line {_lines(bad_fields)}")
    if bad_values:
        failures.append(f"check 3: {len(bad_values)} row(s) with a label_id that is not an integer or a value "
                        f"other than t/f: line {_lines(bad_values)}")
    return (None if failures else rows), failures


def check(flags_path, pool_path, key_path, unmasked_path, schemas, max_missing=0.01):
    """``(failures, summary)``: the named failures (empty if the flags pass) and the counts-only summary.

    Every join is on ``(schema, label_id)`` with ``validate=``: label ids are per city, and both cities' ids overlap
    in one flags file.
    """
    rows, failures = read_flags_lines(flags_path)
    if rows is None:
        return failures, None
    flags = pd.DataFrame(rows, columns=["schema", "label_id", "author_excluded"])
    schemas = list(dict.fromkeys(schemas))
    summary = {"author_excluded_sha256": sha256_file(flags_path), "schemas": schemas, "max_missing": max_missing,
               "rows": {}, "flagged_t": {}, "flagged_f": {}, "pool_labels": {}, "pool_labels_missing": {}}

    # 4. exactly the expected schemas
    got = set(flags.schema)
    if got != set(schemas):
        missing, extra = sorted(set(schemas) - got), sorted(got - set(schemas))
        failures.append(f"check 4: schemas in the flags differ from --schemas (missing: {missing or 'none'}; "
                        f"extra: {extra or 'none'})")
    for s in schemas:
        sel = flags.schema == s
        summary["rows"][s] = int(sel.sum())
        summary["flagged_t"][s] = int((sel & flags.author_excluded.eq("t")).sum())
        summary["flagged_f"][s] = int((sel & flags.author_excluded.eq("f")).sum())

    # 5. unique (schema, label_id)
    dup = flags.duplicated(["schema", "label_id"], keep=False)
    if dup.any():
        refs = {short_ref(s, i) for s, i in zip(flags.schema[dup], flags.label_id[dup])}
        failures.append(f"check 5: {len(refs)} (schema, label_id) repeated: {_named(refs)}")
        return failures, summary  # the joins below need unique keys
    fkeys = flags[["schema", "label_id", "author_excluded"]]

    # 6. every pool schema is covered, and per schema few pool labels are missing from the flags
    pool = pd.read_csv(pool_path, usecols=["schema", "label_id"])
    pool["label_id"] = pool.label_id.astype(int)
    uncovered = sorted(set(pool.schema) - set(schemas))
    if uncovered:
        failures.append(f"check 6: the pool has labels of {uncovered}, not in --schemas: the pull does not cover "
                        f"the round")
    pool = pool[pool.schema.isin(schemas)]
    m = pool.merge(fkeys, on=["schema", "label_id"], how="left", validate="one_to_one", indicator=True)
    for s in schemas:
        sel = m.schema == s
        n_pool = int(sel.sum())
        n_miss = int((sel & (m["_merge"] == "left_only")).sum())
        share = n_miss / n_pool if n_pool else 1.0
        summary["pool_labels"][s] = n_pool
        summary["pool_labels_missing"][s] = {"n": n_miss, "share": round(share, 6)}
        if n_pool == 0:
            failures.append(f"check 6: the pool has no labels of {s}")
        elif share >= max_missing:
            failures.append(f"check 6: {n_miss:,} of {n_pool:,} pool labels of {s} ({share:.2%}) are missing from "
                            f"the flags (limit {max_missing:.2%})")

    # 7. whole key: flagged key items == unmasked items, both directions
    key = pd.read_csv(key_path, usecols=["schema", "label_id"])
    key["label_id"] = key.label_id.astype(int)
    key = key.drop_duplicates()  # a shared item may be listed once per rater; the flag is per label
    unmasked = pd.read_csv(unmasked_path, usecols=["label_uid"])
    um = set(unmasked.label_uid.astype(str))
    k = key.merge(fkeys, on=["schema", "label_id"], how="left", validate="one_to_one", indicator=True)
    k["uid"] = k.schema + ":" + k.label_id.astype(str)
    assert k.uid.is_unique
    flagged = set(k.uid[k.author_excluded.eq("t")])
    missing_key = set(k.uid[k["_merge"] == "left_only"])
    as_ref = lambda uids: {short_ref(*u.rsplit(":", 1)) for u in uids}  # noqa: E731
    summary.update({"key_items": int(len(k)), "key_items_flagged": len(flagged),
                    "key_items_missing": len(missing_key), "unmasked_items": len(um)})
    not_in_key = um - set(k.uid)
    if not_in_key:
        failures.append(f"check 7: {len(not_in_key)} unmasked item(s) not in the key: {_named(as_ref(not_in_key))}")
    if flagged - um:
        failures.append(f"check 7: {len(flagged - um)} key item(s) flagged excluded but not in the unmasked list "
                        f"(rated masked; author excluded since?): {_named(as_ref(flagged - um))}")
    um_missing = (um & missing_key)
    if um_missing:
        failures.append(f"check 7: {len(um_missing)} unmasked item(s) missing from the flags: "
                        f"{_named(as_ref(um_missing))}")
    um_f = um & set(k.uid[k.author_excluded.eq("f")])
    if um_f:
        failures.append(f"check 7: {len(um_f)} unmasked item(s) flagged f (author no longer excluded?): "
                        f"{_named(as_ref(um_f))}")
    return failures, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flags", required=True, help="the pulled schema,label_id,author_excluded CSV")
    ap.add_argument("--pool", required=True, help="the round's pinned pool (csv or csv.gz)")
    ap.add_argument("--key", required=True, help="the round's key.csv (all items)")
    ap.add_argument("--unmasked", required=True, help="unmasked.csv: label_uid of items rated unmasked")
    ap.add_argument("--schemas", nargs="+", required=True)
    ap.add_argument("--max-missing", type=float, default=0.01,
                    help="largest allowed share of a schema's pool labels missing from the flags (default 0.01)")
    ap.add_argument("--out", help="counts-only JSON summary, written only when every check passes")
    a = ap.parse_args(argv)
    failures, summary = check(a.flags, a.pool, a.key, a.unmasked, a.schemas, a.max_missing)
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    if failures:
        print(f"check_author_excluded: {len(failures)} check(s) failed; flags NOT accepted", file=sys.stderr)
        return 1
    summary["checks_passed"] = True
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
            f.write("\n")
    per = ", ".join(f"{s} {summary['rows'][s]:,} rows ({summary['flagged_t'][s]:,} t), "
                    f"{summary['pool_labels_missing'][s]['n']:,} pool labels missing" for s in summary["schemas"])
    print(f"check_author_excluded: all checks passed. {per}; key items flagged {summary['key_items_flagged']} "
          f"(= the {summary['unmasked_items']} unmasked), key items missing {summary['key_items_missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
