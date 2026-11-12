"""Tests for analysis/check_author_excluded.py, the acceptance check on an author-exclusion pull (#85).

Synthetic fixtures only: two cities whose label ids overlap (1..200 in both), so every check has to key on
(schema, label_id). One passing file, then one failing case per check."""
import doctest
import json
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

from analysis import check_author_excluded as cae

SCHEMAS = ["sidewalk_a", "sidewalk_b"]
N = 200
UNMASKED = ["sidewalk_a:5", "sidewalk_a:6"]  # the items rated unmasked: their author is excluded


def _write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return str(path)


def _flag_rows(drop=(), t=("sidewalk_a:5", "sidewalk_a:6", "sidewalk_a:50"), extra=()):
    """Every pool label of both cities plus two newer labels per city; t for the excluded author's labels.
    sidewalk_b:5 and sidewalk_b:6 share their ids with the unmasked items and are f."""
    rows = []
    for s in SCHEMAS:
        for i in range(1, N + 3):
            uid = f"{s}:{i}"
            if uid in drop:
                continue
            rows.append(f"{s},{i},{'t' if uid in t else 'f'}")
    return rows + list(extra)


@pytest.fixture
def files(tmp_path):
    pool = pd.DataFrame([{"schema": s, "label_id": i, "label_type": "Obstacle"} for s in SCHEMAS
                         for i in range(1, N + 1)])
    pool.to_csv(tmp_path / "pool.csv.gz", index=False)
    key = pd.DataFrame([{"schema": s, "label_id": i, "label_uid": f"{s}:{i}", "rater": r}
                        for s, r in zip(SCHEMAS, ["ra", "rb"]) for i in (1, 2, 3, 4, 5, 6, 7, 8)])
    key = pd.concat([key, key.iloc[[0]].assign(rater="rb")])  # a shared item listed once per rater
    key.to_csv(tmp_path / "key.csv", index=False)
    pd.DataFrame({"packet_id": ["p1", "p2"], "label_uid": UNMASKED, "reason": "author excluded"}).to_csv(
        tmp_path / "unmasked.csv", index=False)

    def run(rows, header="schema,label_id,author_excluded", schemas=SCHEMAS, out=True):
        flags = _write_lines(tmp_path / "flags.csv", [header] + rows)
        argv = ["--flags", flags, "--pool", str(tmp_path / "pool.csv.gz"), "--key", str(tmp_path / "key.csv"),
                "--unmasked", str(tmp_path / "unmasked.csv"), "--schemas", *schemas]
        out_path = tmp_path / "summary.json"
        if out_path.exists():
            out_path.unlink()
        if out:
            argv += ["--out", str(out_path)]
        return cae.main(argv), out_path

    return run


def test_doctests():
    assert doctest.testmod(cae).failed == 0


def test_a_complete_pull_passes_and_the_summary_has_counts_only(files, capsys):
    rc, out = files(_flag_rows())
    assert rc == 0, capsys.readouterr().err
    s = json.loads(out.read_text())
    assert s["checks_passed"] is True
    assert s["rows"] == {"sidewalk_a": N + 2, "sidewalk_b": N + 2}
    assert s["flagged_t"] == {"sidewalk_a": 3, "sidewalk_b": 0}
    assert s["pool_labels_missing"]["sidewalk_a"] == {"n": 0, "share": 0.0}
    assert s["key_items"] == 16 and s["key_items_flagged"] == 2 and s["key_items_missing"] == 0
    assert s["author_excluded_sha256"] == cae.sha256_file(str(out.parent / "flags.csv"))
    text = out.read_text()
    assert "label_id" not in text and not re.search(r"sidewalk_[a-z]+:\d", text) and not re.search(r"\b[a-z]+:\d", text)


def test_a_few_missing_pool_labels_pass_and_are_counted(files):
    rc, out = files(_flag_rows(drop={"sidewalk_b:100"}))  # 0.5% < 1%
    assert rc == 0
    assert json.loads(out.read_text())["pool_labels_missing"]["sidewalk_b"] == {"n": 1, "share": 0.005}


def _fails(files, capsys, rows, pattern, **kw):
    rc, out = files(rows, **kw)
    err = capsys.readouterr().err
    assert rc == 1
    assert re.search(pattern, err), err
    assert not out.exists()  # nothing written for a rejected pull
    return err


def test_check1_wrong_header(files, capsys):
    _fails(files, capsys, _flag_rows(), r"check 1: header", header="schema,label_id,excluded")


def test_check2_four_field_row(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_a,999,f,x"]), r"check 2: 1 row\(s\) without exactly 3 fields")


def test_check2_copy_tag_line(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["COPY 404"]), r"check 2: .*line 406")


def test_check3_value_true(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_a,999,true"]), r"check 3")


def test_check3_label_id_not_an_integer(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_a,12a,f"]), r"check 3")


def test_check1_not_utf8(files, capsys, tmp_path):
    rc, out = files(_flag_rows())
    with open(tmp_path / "flags.csv", "ab") as f:
        f.write(b"sidewalk_a,999,\xff\n")
    argv = ["--flags", str(tmp_path / "flags.csv"), "--pool", str(tmp_path / "pool.csv.gz"), "--key",
            str(tmp_path / "key.csv"), "--unmasked", str(tmp_path / "unmasked.csv"), "--schemas", *SCHEMAS]
    capsys.readouterr()
    assert cae.main(argv) == 1
    assert "check 1: the file is not UTF-8" in capsys.readouterr().err


def test_check1_and_3_carriage_returns(files, capsys, tmp_path):
    """psql on the host writes bare newlines; a CRLF file or one stray CR is not what was pulled."""
    crlf = [r + "\r" for r in _flag_rows()]
    _fails(files, capsys, crlf, r"check 1: header", header="schema,label_id,author_excluded\r")
    rows = _flag_rows()
    rows[3] += "\r"
    _fails(files, capsys, rows, r"check 3: 1 row\(s\).*line 5")


def test_check3_label_id_beyond_int64(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_a,99999999999999999999999,f"]), r"check 3")


def test_check4_and_6_missing_schema(files, capsys):
    rows = [r for r in _flag_rows() if not r.startswith("sidewalk_b,")]
    err = _fails(files, capsys, rows, r"check 4: .*missing: \['sidewalk_b'\]")
    assert re.search(r"check 6: 200 of 200 pool labels of sidewalk_b \(100\.00%\)", err)


def test_check4_extra_schema(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_c,1,f"]), r"check 4: .*extra: \['sidewalk_c'\]")


def test_check5_repeated_key(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_b,7,f"]), r"check 5: 1 \(schema, label_id\) repeated: b:7")


def test_check6_two_percent_of_one_city_missing(files, capsys):
    drop = {f"sidewalk_b:{i}" for i in range(100, 104)}  # 4 of 200 = 2%
    _fails(files, capsys, _flag_rows(drop=drop), r"check 6: 4 of 200 pool labels of sidewalk_b \(2\.00%\)")


def test_check6_exactly_the_limit_fails(files, capsys):
    """"Under 1%" excludes 1% itself: 2 of 200."""
    _fails(files, capsys, _flag_rows(drop={"sidewalk_b:100", "sidewalk_b:101"}),
           r"check 6: 2 of 200 pool labels of sidewalk_b \(1\.00%\)")


def test_check6_a_pool_city_left_out_of_schemas(files, capsys):
    """--schemas is what the pull was asked for. A pull of one city, checked against that one city, is consistent
    with itself and still does not cover the round: only the pool can show it."""
    rows = [r for r in _flag_rows() if r.startswith("sidewalk_a,")]
    _fails(files, capsys, rows, r"check 6: the pool has labels of \['sidewalk_b'\], not in --schemas",
           schemas=["sidewalk_a"])


def test_check6_a_schema_with_no_pool_labels(files, capsys):
    _fails(files, capsys, _flag_rows(extra=["sidewalk_c,1,f"]), r"check 6: the pool has no labels of sidewalk_c",
           schemas=SCHEMAS + ["sidewalk_c"])


def test_check7_unmasked_item_not_in_the_key(files, capsys, tmp_path):
    files(_flag_rows())  # writes the fixtures; then list an item the key does not hold as unmasked
    pd.DataFrame({"label_uid": UNMASKED + ["sidewalk_b:150"]}).to_csv(tmp_path / "unmasked.csv", index=False)
    capsys.readouterr()
    _fails(files, capsys, _flag_rows(), r"check 7: 1 unmasked item\(s\) not in the key: b:150")


def test_check7_key_item_flagged_but_not_unmasked(files, capsys):
    err = _fails(files, capsys, _flag_rows(t={"sidewalk_a:5", "sidewalk_a:6", "sidewalk_b:6"}),
                 r"check 7: 1 key item\(s\) flagged excluded but not in the unmasked list.*: b:6")
    assert "a:6" not in err.split("unmasked list")[1]  # sidewalk_a:6 is unmasked; ids overlap across cities


def test_check7_unmasked_item_flagged_f(files, capsys):
    _fails(files, capsys, _flag_rows(t={"sidewalk_a:5"}), r"check 7: 1 unmasked item\(s\) flagged f.*: a:6")


def test_check7_unmasked_item_missing(files, capsys):
    _fails(files, capsys, _flag_rows(drop={"sidewalk_a:6"}), r"check 7: 1 unmasked item\(s\) missing from the flags: a:6")


def test_cli_runs_as_a_script(files, tmp_path):
    files(_flag_rows(), out=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "analysis", "check_author_excluded.py"),
                        "--flags", str(tmp_path / "flags.csv"), "--pool", str(tmp_path / "pool.csv.gz"),
                        "--key", str(tmp_path / "key.csv"), "--unmasked", str(tmp_path / "unmasked.csv"),
                        "--schemas", *SCHEMAS], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "all checks passed" in r.stdout
